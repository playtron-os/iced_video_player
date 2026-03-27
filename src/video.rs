use crate::Error;
use gstreamer as gst;
use gstreamer_allocators::DmaBufMemory;
use gstreamer_app as gst_app;
use gstreamer_app::prelude::*;
use gstreamer_video::VideoMeta;
use iced::widget::image as img;
use iced_wgpu::wgpu;
use std::num::NonZeroU8;
use std::ops::{Deref, DerefMut};
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Video pixel format for the decoded frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VideoFormat {
    /// 8-bit NV12 (4:2:0, 1.5 bytes/pixel).
    Nv12,
    /// 10-bit P010 (4:2:0, 3 bytes/pixel, 16-bit samples).
    P010,
}

impl VideoFormat {
    /// Detect format from a GStreamer caps structure.
    pub fn from_caps(s: &gst::StructureRef) -> Self {
        if let Ok(fmt) = s.get::<&str>("format")
            && fmt.starts_with("P010")
        {
            return Self::P010;
        }
        // DMA-BUF caps may use drm-format instead of format.
        if let Ok(drm_fmt) = s.get::<&str>("drm-format")
            && drm_fmt.starts_with("P010")
        {
            return Self::P010;
        }
        Self::Nv12
    }

    /// wgpu texture format for the Y (luma) plane.
    pub fn y_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Nv12 => wgpu::TextureFormat::R8Unorm,
            Self::P010 => wgpu::TextureFormat::R16Unorm,
        }
    }

    /// wgpu texture format for the UV (chroma) plane.
    pub fn uv_format(self) -> wgpu::TextureFormat {
        match self {
            Self::Nv12 => wgpu::TextureFormat::Rg8Unorm,
            Self::P010 => wgpu::TextureFormat::Rg16Unorm,
        }
    }
}

/// Position in the media.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Position {
    /// Position based on time.
    ///
    /// Not the most accurate format for videos.
    Time(Duration),
    /// Position based on nth frame.
    Frame(u64),
}

impl From<Position> for gst::GenericFormattedValue {
    fn from(pos: Position) -> Self {
        match pos {
            Position::Time(t) => gst::ClockTime::from_nseconds(t.as_nanos() as _).into(),
            Position::Frame(f) => gst::format::Default::from_u64(f).into(),
        }
    }
}

impl From<Duration> for Position {
    fn from(t: Duration) -> Self {
        Position::Time(t)
    }
}

impl From<u64> for Position {
    fn from(f: u64) -> Self {
        Position::Frame(f)
    }
}

#[derive(Debug)]
pub(crate) struct Frame {
    sample: gst::Sample,
    /// When this frame was received from GStreamer.
    pub received_at: Instant,
}

impl Frame {
    pub fn empty() -> Self {
        Self {
            sample: gst::Sample::builder().build(),
            received_at: Instant::now(),
        }
    }

    /// Get a clone of the underlying GStreamer sample (reference-counted).
    pub fn sample(&self) -> gst::Sample {
        self.sample.clone()
    }

    pub fn readable(&'_ self) -> Option<gst::BufferMap<'_, gst::buffer::Readable>> {
        self.sample.buffer().and_then(|x| x.map_readable().ok())
    }

    /// Get the Y-plane stride (line pitch) in bytes from the frame's VideoMeta.
    /// This is critical for proper NV12 decoding, as the stride may differ from width.
    pub fn stride(&self) -> Option<u32> {
        self.sample.buffer().and_then(|buffer| {
            buffer
                .meta::<VideoMeta>()
                .map(|meta| meta.stride()[0] as u32)
        })
    }

    /// Check if this frame's buffer is backed by DMA-BUF memory.
    ///
    /// Returns `true` if at least the first memory block is a DMA-BUF.
    /// For NV12, hardware decoders may provide 1 DMA-BUF (both planes) or
    /// 2 DMA-BUFs (one per plane).
    pub fn is_dmabuf(&self) -> bool {
        self.sample
            .buffer()
            .map(|buf| {
                buf.n_memory() > 0
                    && buf
                        .peek_memory(0)
                        .downcast_memory_ref::<DmaBufMemory>()
                        .is_some()
            })
            .unwrap_or(false)
    }

    /// Extract DMA-BUF file descriptors and plane layout for NV12.
    ///
    /// Returns duplicated file descriptors (caller owns them) together with
    /// stride, offset, and DRM modifier information needed for Vulkan import.
    pub fn dmabuf_fds(&self) -> Option<DmaBufPlanes> {
        let buffer = self.sample.buffer()?;
        let n = buffer.n_memory();
        if n == 0 {
            return None;
        }

        let mem0 = buffer.peek_memory(0);
        let dmabuf0 = mem0.downcast_memory_ref::<DmaBufMemory>()?;
        let fd0 = dmabuf0.fd();

        // Get stride/offset from GstVideoMeta (attached by cudadmabufupload).
        let video_meta = buffer.meta::<VideoMeta>();
        let (y_stride, uv_stride, y_offset, uv_offset) = if let Some(meta) = video_meta {
            let strides = meta.stride();
            let offsets = meta.offset();
            (
                strides.first().copied().unwrap_or(0) as u32,
                strides.get(1).copied().unwrap_or(0) as u32,
                offsets.first().copied().unwrap_or(0) as u32,
                offsets.get(1).copied().unwrap_or(0) as u32,
            )
        } else {
            (0, 0, 0, 0)
        };

        // Parse DRM modifier from caps drm-format field: "NV12:0x..."
        let drm_modifier = self
            .sample
            .caps()
            .and_then(|caps| {
                let s = caps.structure(0)?;
                let drm_fmt = s.get::<&str>("drm-format").ok()?;
                let hex = drm_fmt.split(':').nth(1)?;
                u64::from_str_radix(hex.trim_start_matches("0x"), 16).ok()
            })
            .unwrap_or(0);

        if n >= 2 {
            // Separate DMA-BUF per plane
            let mem1 = buffer.peek_memory(1);
            if let Some(dmabuf1) = mem1.downcast_memory_ref::<DmaBufMemory>() {
                let y_guard = FdGuard::new(unsafe { libc::dup(fd0) })?;
                let uv_guard = FdGuard::new(unsafe { libc::dup(dmabuf1.fd()) })?;
                return Some(DmaBufPlanes {
                    y_fd: y_guard.take(),
                    uv_fd: uv_guard.take(),
                    drm_modifier,
                    y_stride,
                    uv_stride,
                    y_offset,
                    uv_offset,
                });
            }
        }

        // Single DMA-BUF with both planes
        let y_guard = FdGuard::new(unsafe { libc::dup(fd0) })?;
        let uv_guard = FdGuard::new(unsafe { libc::dup(fd0) })?;
        Some(DmaBufPlanes {
            y_fd: y_guard.take(),
            uv_fd: uv_guard.take(),
            drm_modifier,
            y_stride,
            uv_stride,
            y_offset,
            uv_offset,
        })
    }

    /// Debug information about buffer memory types (for logging).
    pub fn debug_memory_info(&self) -> String {
        match self.sample.buffer() {
            Some(buf) => {
                let n = buf.n_memory();
                let mut types = Vec::new();
                for i in 0..n {
                    let mem = buf.peek_memory(i);
                    let is_dmabuf = mem.downcast_memory_ref::<DmaBufMemory>().is_some();
                    types.push(format!("mem[{i}]: dmabuf={is_dmabuf}"));
                }
                format!("n_memory={n}, {}", types.join(", "))
            }
            None => "no buffer".to_string(),
        }
    }
}

/// DMA-BUF file descriptors and layout for NV12/P010 Y and UV planes.
///
/// The fds are duplicated and owned by this struct. Vulkan will take ownership
/// when importing, so they must NOT be closed manually after import.
#[derive(Debug)]
pub(crate) struct DmaBufPlanes {
    pub y_fd: RawFd,
    pub uv_fd: RawFd,
    /// DRM format modifier (0 = linear).
    pub drm_modifier: u64,
    /// Row stride for Y plane in bytes.
    pub y_stride: u32,
    /// Row stride for UV plane in bytes.
    pub uv_stride: u32,
    /// Byte offset of Y plane within its DMA-BUF.
    pub y_offset: u32,
    /// Byte offset of UV plane within its DMA-BUF.
    pub uv_offset: u32,
}

/// RAII wrapper for a raw file descriptor. Closes the fd on drop.
///
/// Use [`.take()`](FdGuard::take) to transfer ownership (e.g., to Vulkan
/// import) without closing.
pub(crate) struct FdGuard(RawFd);

impl FdGuard {
    /// Wrap a duplicated file descriptor. Returns `None` if `fd < 0` (dup failed).
    pub fn new(fd: RawFd) -> Option<Self> {
        if fd < 0 { None } else { Some(Self(fd)) }
    }

    /// Transfer ownership of the fd out of the guard.
    /// The guard will no longer close the fd on drop.
    pub fn take(mut self) -> RawFd {
        let fd = self.0;
        self.0 = -1;
        fd
    }
}

impl Drop for FdGuard {
    fn drop(&mut self) {
        if self.0 >= 0 {
            unsafe { libc::close(self.0) };
        }
    }
}

#[derive(Debug)]
pub(crate) struct Internal {
    pub(crate) id: u64,

    pub(crate) bus: gst::Bus,
    pub(crate) source: gst::Pipeline,
    pub(crate) alive: Arc<AtomicBool>,

    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) framerate: f64,
    pub(crate) duration: Duration,
    pub(crate) format: VideoFormat,
    pub(crate) speed: f64,
    pub(crate) sync_av: bool,

    pub(crate) frame: Arc<Mutex<Frame>>,
    pub(crate) upload_frame: Arc<AtomicBool>,
    /// Set to `true` by update() after publishing on_new_frame.
    /// Cleared by draw() when it consumes upload_frame.
    /// Prevents duplicate NewFrame messages per actual video frame.
    pub(crate) frame_notified: Arc<AtomicBool>,
    pub(crate) looping: bool,
    pub(crate) is_eos: bool,
    pub(crate) restart_stream: bool,
    pub(crate) sync_av_avg: u64,
    pub(crate) sync_av_counter: u64,

    pub(crate) subtitle_text: Arc<Mutex<Option<String>>>,
    pub(crate) upload_text: Arc<AtomicBool>,

    /// Reference to the cudadmabufupload element (if present) for Vulkan export setup.
    pub(crate) cuda_upload: Option<gst::Element>,
}

impl Internal {
    pub(crate) fn seek(&self, position: impl Into<Position>, accurate: bool) -> Result<(), Error> {
        let position = match position.into() {
            // Clamp time-based seeks to duration to avoid GStreamer errors.
            Position::Time(t) if !self.duration.is_zero() && t > self.duration => {
                Position::Time(self.duration)
            }
            other => other,
        };

        // gstreamer complains if the start & end value types aren't the same
        match &position {
            Position::Time(_) => self.source.seek(
                self.speed,
                gst::SeekFlags::FLUSH
                    | if accurate {
                        gst::SeekFlags::ACCURATE
                    } else {
                        gst::SeekFlags::empty()
                    },
                gst::SeekType::Set,
                gst::GenericFormattedValue::from(position),
                gst::SeekType::Set,
                gst::ClockTime::NONE,
            )?,
            Position::Frame(_) => self.source.seek(
                self.speed,
                gst::SeekFlags::FLUSH
                    | if accurate {
                        gst::SeekFlags::ACCURATE
                    } else {
                        gst::SeekFlags::empty()
                    },
                gst::SeekType::Set,
                gst::GenericFormattedValue::from(position),
                gst::SeekType::Set,
                gst::format::Default::NONE,
            )?,
        };

        *self.subtitle_text.lock().expect("lock subtitle_text") = None;
        self.upload_text.store(true, Ordering::SeqCst);

        Ok(())
    }

    pub(crate) fn set_speed(&mut self, speed: f64) -> Result<(), Error> {
        let Some(position) = self.source.query_position::<gst::ClockTime>() else {
            return Err(Error::Caps);
        };
        if speed > 0.0 {
            self.source.seek(
                speed,
                gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE,
                gst::SeekType::Set,
                position,
                gst::SeekType::End,
                gst::ClockTime::from_seconds(0),
            )?;
        } else {
            self.source.seek(
                speed,
                gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE,
                gst::SeekType::Set,
                gst::ClockTime::from_seconds(0),
                gst::SeekType::Set,
                position,
            )?;
        }
        self.speed = speed;
        Ok(())
    }

    pub(crate) fn restart_stream(&mut self) -> Result<(), Error> {
        self.is_eos = false;
        self.set_paused(false);
        self.seek(0, false)?;
        Ok(())
    }

    pub(crate) fn set_paused(&mut self, paused: bool) {
        self.source
            .set_state(if paused {
                gst::State::Paused
            } else {
                gst::State::Playing
            })
            .unwrap(/* state was changed in ctor; state errors caught there */);

        // Set restart_stream flag to make the stream restart on the next Message::NextFrame
        if self.is_eos && !paused {
            self.restart_stream = true;
        }
    }

    pub(crate) fn paused(&self) -> bool {
        self.source.state(gst::ClockTime::ZERO).1 == gst::State::Paused
    }

    /// Syncs audio with video when there is (inevitably) latency presenting the frame.
    pub(crate) fn set_av_offset(&mut self, offset: Duration) {
        if self.sync_av {
            self.sync_av_counter += 1;
            self.sync_av_avg = self.sync_av_avg * (self.sync_av_counter - 1) / self.sync_av_counter
                + offset.as_nanos() as u64 / self.sync_av_counter;
            if self.sync_av_counter.is_multiple_of(128) {
                self.source
                    .set_property("av-offset", -(self.sync_av_avg as i64));
            }
        }
    }
}

/// A multimedia video loaded from a URI (e.g., a local file path or HTTP stream).
#[derive(Debug)]
pub struct Video(pub(crate) RwLock<Internal>);

impl Drop for Video {
    fn drop(&mut self) {
        let inner = self.0.get_mut().expect("failed to lock");

        // Setting the pipeline to Null is synchronous and waits for any
        // currently executing AppSink callbacks to finish.
        inner
            .source
            .set_state(gst::State::Null)
            .expect("failed to set state");

        inner.alive.store(false, Ordering::SeqCst);
    }
}

impl Video {
    /// Create a new video player from a given video which loads from `uri`.
    /// Note that live sources will report the duration to be zero.
    ///
    /// Attempts a DMA-BUF zero-copy pipeline first (no `videoconvert`, hardware
    /// decoders can pass DMA-BUF NV12 directly). Falls back to CPU-copy pipeline
    /// with `videoconvert` if the DMA-BUF pipeline fails to negotiate.
    pub fn new(uri: &url::Url) -> Result<Self, Error> {
        gst::init()?;

        // Pipeline priority:
        // 1. NVIDIA CUDA → DMA-BUF zero-copy (nvh264dec → cudadmabufupload → appsink)
        // 2. Generic DMA-BUF (VA-API decoders produce DMA-BUF directly)
        // 3. CPU fallback (videoconvert → system memory NV12)

        // Try NVIDIA CUDA → DMA-BUF pipeline (cudadmabufupload converts CUDA memory to DMA-BUF)
        let nvidia_pipeline = format!(
            "playbin uri=\"{}\" text-sink=\"appsink name=iced_text sync=true drop=true\" video-sink=\"cudadmabufupload name=cuda_upload force-linear=true ! appsink name=iced_video drop=true\"",
            uri.as_str()
        );
        match Self::try_launch_playbin(&nvidia_pipeline) {
            Ok(video) => {
                info!("Using NVIDIA CUDA→DMA-BUF zero-copy pipeline");
                return Ok(video);
            }
            Err(e) => {
                debug!(error = %e, "NVIDIA DMA-BUF pipeline not available, trying next");
            }
        }

        // Try generic DMA-BUF pipeline (VA-API decoders on Intel/AMD)
        let dmabuf_pipeline = format!(
            "playbin uri=\"{}\" text-sink=\"appsink name=iced_text sync=true drop=true\" video-sink=\"appsink name=iced_video drop=true caps=video/x-raw(memory:DMABuf),format={{NV12,P010_10LE}},pixel-aspect-ratio=1/1;video/x-raw,format={{NV12,P010_10LE}},pixel-aspect-ratio=1/1\"",
            uri.as_str()
        );
        match Self::try_launch_playbin(&dmabuf_pipeline) {
            Ok(video) => {
                info!("Using DMA-BUF-capable pipeline");
                return Ok(video);
            }
            Err(e) => {
                debug!(error = %e, "DMA-BUF pipeline not available, trying next");
            }
        }

        // Fallback: CPU copy with videoconvert
        info!("Falling back to CPU-copy pipeline");
        let cpu_pipeline = format!(
            "playbin uri=\"{}\" text-sink=\"appsink name=iced_text sync=true drop=true\" video-sink=\"videoscale ! videoconvert ! appsink name=iced_video drop=true caps=video/x-raw,format={{NV12,P010_10LE}},pixel-aspect-ratio=1/1\"",
            uri.as_str()
        );
        Self::try_launch_playbin(&cpu_pipeline)
    }

    /// Try to launch a playbin pipeline string and wire up sinks.
    fn try_launch_playbin(pipeline_str: &str) -> Result<Self, Error> {
        let pipeline = gst::parse::launch(pipeline_str)?
            .downcast::<gst::Pipeline>()
            .map_err(|_| Error::Cast)?;

        // Extract the video appsink from the video-sink property.
        // gst_parse wraps element descriptions in a GstBin with ghost pads,
        // so we first try the Bin path, then fall back to direct downcast.
        let video_sink: gst::Element = pipeline.property("video-sink");
        let video_sink = Self::find_appsink(&video_sink, "iced_video")?;

        // Try to find the cudadmabufupload element for Vulkan export setup.
        let cuda_upload = Self::find_element_by_name(&pipeline, "cuda_upload");

        let text_sink: gst::Element = pipeline.property("text-sink");
        let text_sink = Self::find_appsink(&text_sink, "iced_text")?;

        let mut video = Self::from_gst_pipeline(pipeline, video_sink, Some(text_sink))?;
        video.0.get_mut().expect("lock").cuda_upload = cuda_upload;
        Ok(video)
    }

    /// Find a named AppSink inside an element that may be a Bin wrapper.
    fn find_appsink(element: &gst::Element, name: &str) -> Result<gst_app::AppSink, Error> {
        // Direct name match — element itself is the appsink.
        if element.name().as_str() == name {
            return element
                .clone()
                .downcast::<gst_app::AppSink>()
                .map_err(|_| Error::Cast);
        }

        // The element may be a Bin (gst_parse wraps multi-element pipelines).
        if let Ok(bin) = element.clone().downcast::<gst::Bin>()
            && let Some(found) = bin.by_name(name)
        {
            return found
                .downcast::<gst_app::AppSink>()
                .map_err(|_| Error::Cast);
        }

        // Try ghost pad parent — gst_parse wraps single elements too.
        if let Some(pad) = element.pads().first()
            && let Ok(ghost) = pad.clone().dynamic_cast::<gst::GhostPad>()
            && let Some(parent) = ghost.parent_element()
            && let Ok(bin) = parent.downcast::<gst::Bin>()
            && let Some(found) = bin.by_name(name)
        {
            return found
                .downcast::<gst_app::AppSink>()
                .map_err(|_| Error::Cast);
        }

        Err(Error::AppSink(name.to_string()))
    }

    /// Find a named element in the pipeline hierarchy.
    fn find_element_by_name(pipeline: &gst::Pipeline, name: &str) -> Option<gst::Element> {
        // playbin → video-sink may be a Bin
        let video_sink: gst::Element = pipeline.property("video-sink");
        if let Ok(bin) = video_sink.clone().downcast::<gst::Bin>()
            && let Some(found) = bin.by_name(name)
        {
            return Some(found.upcast());
        }
        // Try the pipeline directly (recurse all children)
        pipeline.by_name(name).map(|e| e.upcast())
    }

    /// Creates a new video based on an existing GStreamer pipeline and appsink.
    /// Expects an `appsink` with `caps=video/x-raw,format={NV12,P010_10LE}`.
    ///
    /// An optional `text_sink` can be provided, which enables subtitle messages
    /// to be emitted.
    ///
    /// **Note:** Many functions of [`Video`] assume a `playbin` pipeline.
    /// Non-`playbin` pipelines given here may not have full functionality.
    pub fn from_gst_pipeline(
        pipeline: gst::Pipeline,
        video_sink: gst_app::AppSink,
        text_sink: Option<gst_app::AppSink>,
    ) -> Result<Self, Error> {
        gst::init()?;
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        // We need to ensure we stop the pipeline if we hit an error,
        // or else there may be audio left playing in the background.
        macro_rules! cleanup {
            ($expr:expr) => {
                $expr.map_err(|e| {
                    let _ = pipeline.set_state(gst::State::Null);
                    e
                })
            };
        }

        let pad = video_sink.pads().first().cloned().unwrap();

        cleanup!(pipeline.set_state(gst::State::Playing))?;

        // wait for up to 5 seconds until the decoder gets the source capabilities
        cleanup!(pipeline.state(gst::ClockTime::from_seconds(5)).0)?;

        // extract resolution, framerate, and pixel format
        let caps = cleanup!(pad.current_caps().ok_or(Error::Caps))?;
        let s = cleanup!(caps.structure(0).ok_or(Error::Caps))?;
        let width = cleanup!(s.get::<i32>("width").map_err(|_| Error::Caps))?;
        let height = cleanup!(s.get::<i32>("height").map_err(|_| Error::Caps))?;
        let framerate = cleanup!(s.get::<gst::Fraction>("framerate").map_err(|_| Error::Caps))?;
        let framerate = framerate.numer() as f64 / framerate.denom() as f64;
        let format = VideoFormat::from_caps(s);

        if framerate.is_nan()
            || framerate.is_infinite()
            || framerate < 0.0
            || framerate.abs() < f64::EPSILON
        {
            let _ = pipeline.set_state(gst::State::Null);
            return Err(Error::Framerate(framerate));
        }

        let duration = Duration::from_nanos(
            pipeline
                .query_duration::<gst::ClockTime>()
                .map(|duration| duration.nseconds())
                .unwrap_or(0),
        );

        let sync_av = pipeline.has_property("av-offset", None);

        info!(width, height, ?format, framerate, "Video negotiated");

        let frame = Arc::new(Mutex::new(Frame::empty()));
        let upload_frame = Arc::new(AtomicBool::new(false));
        let alive = Arc::new(AtomicBool::new(true));

        let subtitle_text = Arc::new(Mutex::new(None));
        let upload_text = Arc::new(AtomicBool::new(false));

        // Shared state for subtitle end-time tracking (cleared by video callback).
        let clear_subtitles_at: Arc<Mutex<Option<gst::ClockTime>>> = Arc::new(Mutex::new(None));

        // --- Text sink callback (subtitles) ---
        if let Some(ref text_sink) = text_sink {
            let subtitle_text_ref = Arc::clone(&subtitle_text);
            let upload_text_ref = Arc::clone(&upload_text);
            let clear_at_ref = Arc::clone(&clear_subtitles_at);

            text_sink.set_callbacks(
                gst_app::AppSinkCallbacks::builder()
                    .new_sample(move |sink| {
                        let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                        let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                        let text_pts = buffer.pts().ok_or(gst::FlowError::Error)?;
                        let text_duration = buffer.duration().unwrap_or(gst::ClockTime::ZERO);

                        let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                        let text = std::str::from_utf8(map.as_slice())
                            .map_err(|_| gst::FlowError::Error)?
                            .to_string();

                        if let Ok(mut guard) = subtitle_text_ref.lock() {
                            *guard = Some(text);
                        }
                        upload_text_ref.store(true, Ordering::SeqCst);

                        if let Ok(mut guard) = clear_at_ref.lock() {
                            *guard = Some(text_pts + text_duration);
                        }

                        Ok(gst::FlowSuccess::Ok)
                    })
                    .build(),
            );
        }

        // --- Video sink callbacks (frame delivery) ---
        {
            let frame_ref = Arc::clone(&frame);
            let upload_frame_ref = Arc::clone(&upload_frame);
            let subtitle_text_ref = Arc::clone(&subtitle_text);
            let upload_text_ref = Arc::clone(&upload_text);
            let clear_at_ref = Arc::clone(&clear_subtitles_at);

            let handle_sample = move |sink: &gst_app::AppSink,
                                      is_preroll: bool|
                  -> Result<gst::FlowSuccess, gst::FlowError> {
                let sample = if is_preroll {
                    sink.pull_preroll().map_err(|_| gst::FlowError::Eos)?
                } else {
                    sink.pull_sample().map_err(|_| gst::FlowError::Eos)?
                };

                // Check if current frame is past subtitle end time → clear subtitle.
                if let Some(buffer) = sample.buffer()
                    && let Some(frame_pts) = buffer.pts()
                    && let Ok(mut clear_guard) = clear_at_ref.lock()
                    && let Some(at) = *clear_guard
                    && frame_pts >= at
                {
                    if let Ok(mut sub) = subtitle_text_ref.lock() {
                        *sub = None;
                    }
                    upload_text_ref.store(true, Ordering::SeqCst);
                    *clear_guard = None;
                }

                if let Ok(mut frame_guard) = frame_ref.lock() {
                    *frame_guard = Frame {
                        sample,
                        received_at: Instant::now(),
                    };
                }
                upload_frame_ref.store(true, Ordering::SeqCst);

                Ok(gst::FlowSuccess::Ok)
            };

            let handle_sample_new = handle_sample.clone();
            let handle_sample_preroll = handle_sample;

            video_sink.set_callbacks(
                gst_app::AppSinkCallbacks::builder()
                    .new_sample(move |sink| handle_sample_new(sink, false))
                    .new_preroll(move |sink| handle_sample_preroll(sink, true))
                    .build(),
            );
        }

        Ok(Video(RwLock::new(Internal {
            id,

            bus: pipeline.bus().unwrap(),
            source: pipeline,
            alive,

            width,
            height,
            framerate,
            duration,
            format,
            speed: 1.0,
            sync_av,

            frame,
            upload_frame,
            frame_notified: Arc::new(AtomicBool::new(false)),
            looping: false,
            is_eos: false,
            restart_stream: false,
            sync_av_avg: 0,
            sync_av_counter: 0,

            subtitle_text,
            upload_text,

            cuda_upload: None,
        })))
    }

    pub(crate) fn read(&self) -> impl Deref<Target = Internal> + '_ {
        self.0.read().expect("lock")
    }

    pub(crate) fn write(&self) -> impl DerefMut<Target = Internal> + '_ {
        self.0.write().expect("lock")
    }

    pub(crate) fn get_mut(&mut self) -> impl DerefMut<Target = Internal> + '_ {
        self.0.get_mut().expect("lock")
    }

    /// Get the size/resolution of the video as `(width, height)`.
    pub fn size(&self) -> (i32, i32) {
        (self.read().width, self.read().height)
    }

    /// Get the framerate of the video as frames per second.
    pub fn framerate(&self) -> f64 {
        self.read().framerate
    }

    /// Set the volume multiplier of the audio.
    /// `0.0` = 0% volume, `1.0` = 100% volume.
    ///
    /// This uses a linear scale, for example `0.5` is perceived as half as loud.
    pub fn set_volume(&mut self, volume: f64) {
        self.get_mut().source.set_property("volume", volume);
        self.set_muted(self.muted()); // for some reason gstreamer unmutes when changing volume?
    }

    /// Get the volume multiplier of the audio.
    pub fn volume(&self) -> f64 {
        self.read().source.property("volume")
    }

    /// Set if the audio is muted or not, without changing the volume.
    pub fn set_muted(&mut self, muted: bool) {
        self.get_mut().source.set_property("mute", muted);
    }

    /// Get if the audio is muted or not.
    pub fn muted(&self) -> bool {
        self.read().source.property("mute")
    }

    /// Get if the stream ended or not.
    pub fn eos(&self) -> bool {
        self.read().is_eos
    }

    /// Get if the media will loop or not.
    pub fn looping(&self) -> bool {
        self.read().looping
    }

    /// Set if the media will loop or not.
    pub fn set_looping(&mut self, looping: bool) {
        self.get_mut().looping = looping;
    }

    /// Set if the media is paused or not.
    pub fn set_paused(&mut self, paused: bool) {
        self.get_mut().set_paused(paused)
    }

    /// Get if the media is paused or not.
    pub fn paused(&self) -> bool {
        self.read().paused()
    }

    /// Jumps to a specific position in the media.
    /// Passing `true` to the `accurate` parameter will result in more accurate seeking,
    /// however, it is also slower. For most seeks (e.g., scrubbing) this is not needed.
    pub fn seek(&mut self, position: impl Into<Position>, accurate: bool) -> Result<(), Error> {
        self.get_mut().seek(position, accurate)
    }

    /// Steps forward exactly one frame in playback.
    /// This can be especially useful while the video is paused to make pipeline changes visible, without resuming playback.
    pub fn step_one_frame(&mut self) {
        self.get_mut().source.send_event(gst::event::Step::new(
            gst::GenericFormattedValue::Buffers(Some(gst::format::Buffers::from_u64(1))),
            1.0,
            true,
            false,
        ));
    }

    /// Set the playback speed of the media.
    /// The default speed is `1.0`.
    pub fn set_speed(&mut self, speed: f64) -> Result<(), Error> {
        self.get_mut().set_speed(speed)
    }

    /// Get the current playback speed.
    pub fn speed(&self) -> f64 {
        self.read().speed
    }

    /// Get the current playback position in time.
    pub fn position(&self) -> Duration {
        Duration::from_nanos(
            self.read()
                .source
                .query_position::<gst::ClockTime>()
                .map_or(0, |pos| pos.nseconds()),
        )
    }

    /// Get the media duration.
    pub fn duration(&self) -> Duration {
        self.read().duration
    }

    /// Restarts a stream; seeks to the first frame and unpauses, sets the `eos` flag to false.
    pub fn restart_stream(&mut self) -> Result<(), Error> {
        self.get_mut().restart_stream()
    }

    /// Set the subtitle URL to display.
    pub fn set_subtitle_url(&mut self, url: &url::Url) -> Result<(), Error> {
        let paused = self.paused();
        let mut inner = self.get_mut();
        inner.source.set_state(gst::State::Ready)?;
        inner.source.set_property("suburi", url.as_str());
        inner.set_paused(paused);
        Ok(())
    }

    /// Get the current subtitle URL.
    pub fn subtitle_url(&self) -> Option<url::Url> {
        url::Url::parse(
            &self
                .read()
                .source
                .property::<Option<String>>("current-suburi")?,
        )
        .ok()
    }

    /// Get the underlying GStreamer pipeline.
    pub fn pipeline(&self) -> gst::Pipeline {
        self.read().source.clone()
    }

    /// Generates a list of thumbnails based on a set of positions in the media, downscaled by a given factor.
    ///
    /// Slow; only needs to be called once for each instance.
    /// It's best to call this at the very start of playback, otherwise the position may shift.
    pub fn thumbnails<I>(
        &mut self,
        positions: I,
        downscale: NonZeroU8,
    ) -> Result<Vec<img::Handle>, Error>
    where
        I: IntoIterator<Item = Position>,
    {
        let downscale = u8::from(downscale) as u32;

        let paused = self.paused();
        let muted = self.muted();
        let pos = self.position();

        self.set_paused(false);
        self.set_muted(true);

        let out = {
            let inner = self.read();
            let width = inner.width;
            let height = inner.height;
            positions
                .into_iter()
                .map(|pos| {
                    inner.seek(pos, true)?;
                    inner.upload_frame.store(false, Ordering::SeqCst);
                    let deadline = Instant::now() + Duration::from_secs(5);
                    while !inner.upload_frame.load(Ordering::SeqCst) {
                        if Instant::now() >= deadline {
                            return Err(Error::Timeout);
                        }
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    let frame_guard = inner.frame.lock().map_err(|_| Error::Lock)?;
                    let frame = frame_guard.readable().ok_or(Error::Lock)?;
                    let stride = frame_guard.stride();

                    Ok(img::Handle::from_rgba(
                        inner.width as u32 / downscale,
                        inner.height as u32 / downscale,
                        yuv_to_rgba(
                            frame.as_slice(),
                            width as _,
                            height as _,
                            downscale,
                            stride,
                            inner.format,
                        ),
                    ))
                })
                .collect()
        };

        self.set_paused(paused);
        self.set_muted(muted);
        self.seek(pos, true)?;

        out
    }
}

fn yuv_to_rgba(
    yuv: &[u8],
    width: u32,
    height: u32,
    downscale: u32,
    stride: Option<u32>,
    format: VideoFormat,
) -> Vec<u8> {
    // Use stride from VideoMeta if available, otherwise assume stride == width
    // (for P010 the stride is in bytes and already accounts for 2 bytes/sample).
    let stride = stride.unwrap_or(match format {
        VideoFormat::Nv12 => width,
        VideoFormat::P010 => width * 2,
    });

    let uv_start = stride * height;
    let out_w = (width / downscale) as usize;
    let out_h = (height / downscale) as usize;

    // Validate buffer size.
    // NV12: stride*height (Y) + stride*height/2 (UV) = stride*height*1.5
    // P010: stride*height (Y, 2 bytes/sample) + stride*height/2 (UV, 2 bytes/sample) = stride*height*1.5
    // Both have the same ratio because stride already accounts for sample size.
    let required = (uv_start + stride * height / 2) as usize;
    if yuv.len() < required {
        tracing::warn!(
            buffer_len = yuv.len(),
            required,
            ?format,
            "truncated buffer, returning black frame"
        );
        return vec![0; out_w * out_h * 4];
    }

    let mut rgba = Vec::with_capacity(out_w * out_h * 4);

    for row in 0..height / downscale {
        for col in 0..width / downscale {
            let x_src = col * downscale;
            let y_src = row * downscale;

            let (y_val, u_val, v_val) = match format {
                VideoFormat::Nv12 => {
                    // NV12: 1 byte per Y sample, interleaved UV (2 bytes per 2×2 block)
                    let y_off = (y_src * stride + x_src) as usize;
                    let uv_off = (uv_start + (y_src / 2) * stride + (x_src / 2) * 2) as usize;
                    (
                        yuv[y_off] as f32,
                        yuv[uv_off] as f32,
                        yuv[uv_off + 1] as f32,
                    )
                }
                VideoFormat::P010 => {
                    // P010: 2 bytes (u16 LE) per Y sample, interleaved UV (4 bytes per 2×2 block)
                    let y_off = (y_src * stride + x_src * 2) as usize;
                    let uv_off = (uv_start + (y_src / 2) * stride + (x_src / 2) * 4) as usize;
                    // Read 16-bit little-endian values, shift right 6 to get 10-bit range (0-1023)
                    let y16 = u16::from_le_bytes([yuv[y_off], yuv[y_off + 1]]) >> 6;
                    let u16v = u16::from_le_bytes([yuv[uv_off], yuv[uv_off + 1]]) >> 6;
                    let v16 = u16::from_le_bytes([yuv[uv_off + 2], yuv[uv_off + 3]]) >> 6;
                    // Scale 10-bit (0-1023) to 8-bit range (0-255) for the same BT.601 math
                    (
                        (y16 as f32) * 255.0 / 1023.0,
                        (u16v as f32) * 255.0 / 1023.0,
                        (v16 as f32) * 255.0 / 1023.0,
                    )
                }
            };

            // BT.601 narrow-range YUV→RGB (8-bit scale)
            let r = 1.164 * (y_val - 16.0) + 1.596 * (v_val - 128.0);
            let g = 1.164 * (y_val - 16.0) - 0.813 * (v_val - 128.0) - 0.391 * (u_val - 128.0);
            let b = 1.164 * (y_val - 16.0) + 2.018 * (u_val - 128.0);

            rgba.push(r.clamp(0.0, 255.0) as u8);
            rgba.push(g.clamp(0.0, 255.0) as u8);
            rgba.push(b.clamp(0.0, 255.0) as u8);
            rgba.push(0xFF);
        }
    }

    rgba
}
