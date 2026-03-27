use crate::video::{DmaBufPlanes, FdGuard, Frame, VideoFormat};
use gstreamer as gst;
use gstreamer::prelude::*;
use iced_wgpu::primitive::{Pipeline, Primitive};
use iced_wgpu::wgpu;
use std::{
    collections::{BTreeMap, btree_map::Entry},
    num::NonZero,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use tracing::{info, trace, warn};

/// Exported file descriptors for a video's Y and UV plane textures.
///
/// These FDs represent Vulkan-allocated memory that can be imported by CUDA
/// (via `cuImportExternalMemory`) for direct writes. The caller is responsible
/// for closing the FDs when done.
#[derive(Debug)]
pub struct ExportedPlanes {
    /// Opaque file descriptor for the Y (luma) plane.
    pub y_fd: std::os::unix::io::RawFd,
    /// Allocation size of the Y plane in bytes.
    pub y_size: u64,
    /// Row pitch (stride) of the Y plane in bytes.
    pub y_stride: u64,
    /// Opaque file descriptor for the UV (chroma) plane.
    pub uv_fd: std::os::unix::io::RawFd,
    /// Allocation size of the UV plane in bytes.
    pub uv_size: u64,
    /// Row pitch (stride) of the UV plane in bytes.
    pub uv_stride: u64,
}

#[repr(C)]
struct Uniforms {
    rect: [f32; 4],
    // because wgpu min_uniform_buffer_offset_alignment
    _pad: [u8; 240],
}

struct VideoEntry {
    texture_y: wgpu::Texture,
    texture_uv: wgpu::Texture,
    instances: wgpu::Buffer,
    bg0: wgpu::BindGroup,
    alive: Arc<AtomicBool>,

    prepare_index: AtomicUsize,
    render_index: AtomicUsize,

    /// Holds the previous frame's GStreamer sample to prevent the DMA-BUF
    /// GBM buffer from being recycled by gst-cuda-dmabuf before the GPU
    /// finishes copying from it. Cleared when the next frame arrives.
    _prev_dmabuf_sample: Option<gst::Sample>,

    /// When true, CUDA writes directly into these Vulkan textures via exported FDs.
    /// No per-frame import or copy is needed — just render the existing textures.
    using_vulkan_export: bool,
}

pub(crate) struct VideoPipeline {
    pipeline: wgpu::RenderPipeline,
    bg0_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    videos: BTreeMap<u64, VideoEntry>,
}

impl Pipeline for VideoPipeline {
    fn new(device: &wgpu::Device, _queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iced_video_player shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bg0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iced_video_player bind group 0 layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iced_video_player pipeline layout"),
            bind_group_layouts: &[&bg0_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("iced_video_player pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("iced_video_player sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        VideoPipeline {
            pipeline,
            bg0_layout,
            sampler,
            videos: BTreeMap::new(),
        }
    }

    fn trim(&mut self) {
        let ids: Vec<_> = self
            .videos
            .iter()
            .filter_map(|(id, entry)| (!entry.alive.load(Ordering::SeqCst)).then_some(*id))
            .collect();
        for id in ids {
            if let Some(video) = self.videos.remove(&id) {
                video.texture_y.destroy();
                video.texture_uv.destroy();
                video.instances.destroy();
            }
        }
    }
}

impl VideoPipeline {
    #[allow(clippy::too_many_arguments)]
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        video_id: u64,
        alive: &Arc<AtomicBool>,
        (width, height): (u32, u32),
        format: VideoFormat,
        frame: &[u8],
        stride: Option<u32>,
    ) {
        // Use stride from GStreamer's VideoMeta if available.
        // Stride is in bytes: NV12 = 1 byte/pixel, P010 = 2 bytes/pixel.
        let stride = stride.unwrap_or(match format {
            VideoFormat::Nv12 => width,
            VideoFormat::P010 => width * 2,
        });
        if let Entry::Vacant(entry) = self.videos.entry(video_id) {
            let texture_y = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("iced_video_player texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: format.y_format(),
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let texture_uv = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("iced_video_player texture"),
                size: wgpu::Extent3d {
                    width: width / 2,
                    height: height / 2,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: format.uv_format(),
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let view_y = texture_y.create_view(&wgpu::TextureViewDescriptor {
                label: Some("iced_video_player texture view"),
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
                usage: None,
            });

            let view_uv = texture_uv.create_view(&wgpu::TextureViewDescriptor {
                label: Some("iced_video_player texture view"),
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
                usage: None,
            });

            let instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iced_video_player uniform buffer"),
                size: 256 * std::mem::size_of::<Uniforms>() as u64, // max 256 video players per frame
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iced_video_player bind group"),
                layout: &self.bg0_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_y),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_uv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &instances,
                            offset: 0,
                            size: Some(NonZero::new(std::mem::size_of::<Uniforms>() as _).unwrap()),
                        }),
                    },
                ],
            });

            entry.insert(VideoEntry {
                texture_y,
                texture_uv,
                instances,
                bg0: bind_group,
                alive: Arc::clone(alive),

                prepare_index: AtomicUsize::new(0),
                render_index: AtomicUsize::new(0),
                _prev_dmabuf_sample: None,
                using_vulkan_export: false,
            });
        }

        let VideoEntry {
            texture_y,
            texture_uv,
            ..
        } = self.videos.get(&video_id).unwrap();

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: texture_y,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame[..(stride * height) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: texture_uv,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame[(stride * height) as usize..],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(height / 2),
            },
            wgpu::Extent3d {
                width: width / 2,
                height: height / 2,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Import DMA-BUF file descriptors as GPU textures (zero-copy path).
    ///
    /// Creates new wgpu textures backed by the DMA-BUF memory and rebuilds the
    /// bind group. The old textures are destroyed. Vulkan takes ownership of the
    /// duplicated fds.
    ///
    /// Returns `true` if the import succeeded, `false` if it failed (caller
    /// should fall back to the CPU `upload()` path).
    #[allow(clippy::too_many_arguments)]
    fn upload_dmabuf(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        video_id: u64,
        alive: &Arc<AtomicBool>,
        (width, height): (u32, u32),
        format: VideoFormat,
        planes: DmaBufPlanes,
        sample: gst::Sample,
    ) -> bool {
        use iced_wgpu::wgpu::hal::vulkan as vk_hal;

        // FdGuard ensures fds are closed on any early return.
        // .take() transfers ownership to Vulkan import below.
        let y_guard = FdGuard::new(planes.y_fd).expect("y_fd from DmaBufPlanes must be valid");
        let uv_guard = FdGuard::new(planes.uv_fd).expect("uv_fd from DmaBufPlanes must be valid");

        // Imported DMA-BUF textures need COPY_SRC so we can copy to GPU-local textures.
        let import_y_desc = wgpu::TextureDescriptor {
            label: Some("iced_video_player dmabuf Y (import)"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.y_format(),
            usage: wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        let import_uv_desc = wgpu::TextureDescriptor {
            label: Some("iced_video_player dmabuf UV (import)"),
            size: wgpu::Extent3d {
                width: width / 2,
                height: height / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.uv_format(),
            usage: wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };

        // GPU-local textures that persist for rendering.
        let gpu_y_desc = wgpu::TextureDescriptor {
            label: Some("iced_video_player Y"),
            size: import_y_desc.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.y_format(),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let gpu_uv_desc = wgpu::TextureDescriptor {
            label: Some("iced_video_player UV"),
            size: import_uv_desc.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.uv_format(),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        // Convert wgpu TextureDescriptor to HAL TextureDescriptor for import.
        let hal_y_desc = wgpu::hal::TextureDescriptor {
            label: import_y_desc.label,
            size: import_y_desc.size,
            mip_level_count: import_y_desc.mip_level_count,
            sample_count: import_y_desc.sample_count,
            dimension: import_y_desc.dimension,
            format: import_y_desc.format,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };

        let hal_uv_desc = wgpu::hal::TextureDescriptor {
            label: import_uv_desc.label,
            size: import_uv_desc.size,
            mip_level_count: import_uv_desc.mip_level_count,
            sample_count: import_uv_desc.sample_count,
            dimension: import_uv_desc.dimension,
            format: import_uv_desc.format,
            usage: wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };

        let y_plane_info = vk_hal::DmaBufPlaneInfo {
            drm_modifier: planes.drm_modifier,
            stride: planes.y_stride,
            offset: planes.y_offset,
            total_size: 0, // dedicated allocation at offset 0
        };

        let uv_plane_info = vk_hal::DmaBufPlaneInfo {
            drm_modifier: planes.drm_modifier,
            stride: planes.uv_stride,
            offset: planes.uv_offset,
            total_size: if planes.uv_offset > 0 {
                // UV starts at an offset — need to import the full buffer
                (planes.uv_offset as u64) + (planes.uv_stride as u64) * (height as u64 / 2)
            } else {
                0
            },
        };

        trace!(
            drm_modifier = format_args!("0x{:x}", planes.drm_modifier),
            y_stride = planes.y_stride,
            uv_stride = planes.uv_stride,
            y_offset = planes.y_offset,
            uv_offset = planes.uv_offset,
            "DMA-BUF import"
        );

        // Import DMA-BUF fds via the Vulkan HAL.
        // Safety: we pass duplicated fds; Vulkan takes ownership. The DMA-BUF
        // memory remains valid while the GStreamer buffer is held by Frame.
        let (imported_y, imported_uv) = unsafe {
            let hal_device_guard = match device.as_hal::<vk_hal::Api>() {
                Some(guard) => guard,
                None => {
                    warn!("DMA-BUF import unavailable: not a Vulkan backend");
                    // y_guard and uv_guard drop here, closing both fds.
                    return false;
                }
            };

            // .take() transfers fd ownership to Vulkan; on failure Vulkan
            // consumes it internally, so FdGuard must not close it.
            let hal_tex_y = match hal_device_guard.texture_from_dmabuf_fd(
                y_guard.take(),
                &hal_y_desc,
                &y_plane_info,
            ) {
                Ok(t) => t,
                Err(e) => {
                    warn!(error = ?e, "DMA-BUF Y plane import failed");
                    // uv_guard drops here, closing uv_fd.
                    return false;
                }
            };

            let hal_tex_uv = match hal_device_guard.texture_from_dmabuf_fd(
                uv_guard.take(),
                &hal_uv_desc,
                &uv_plane_info,
            ) {
                Ok(t) => t,
                Err(e) => {
                    warn!(error = ?e, "DMA-BUF UV plane import failed");
                    // uv_fd was consumed; hal_tex_y will leak but that's acceptable for error path
                    return false;
                }
            };

            // Wrap HAL textures into wgpu public Texture objects.
            let imported_y =
                device.create_texture_from_hal::<vk_hal::Api>(hal_tex_y, &import_y_desc);
            let imported_uv =
                device.create_texture_from_hal::<vk_hal::Api>(hal_tex_uv, &import_uv_desc);

            (imported_y, imported_uv)
        };

        // Create GPU-local textures and immediately copy the DMA-BUF data.
        // This captures the frame data before gst-cuda-dmabuf can recycle the
        // GBM buffer for the next frame.
        let texture_y = device.create_texture(&gpu_y_desc);
        let texture_uv = device.create_texture(&gpu_uv_desc);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("iced_video_player dmabuf copy"),
        });
        encoder.copy_texture_to_texture(
            imported_y.as_image_copy(),
            texture_y.as_image_copy(),
            import_y_desc.size,
        );
        encoder.copy_texture_to_texture(
            imported_uv.as_image_copy(),
            texture_uv.as_image_copy(),
            import_uv_desc.size,
        );
        queue.submit(Some(encoder.finish()));

        // Imported DMA-BUF textures can now be dropped; wgpu keeps them alive
        // until the copy command completes on the GPU.
        drop(imported_y);
        drop(imported_uv);

        let view_y = texture_y.create_view(&wgpu::TextureViewDescriptor::default());
        let view_uv = texture_uv.create_view(&wgpu::TextureViewDescriptor::default());

        // Ensure the VideoEntry exists (create uniform buffer if first time).
        if let Entry::Vacant(entry) = self.videos.entry(video_id) {
            let instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iced_video_player uniform buffer"),
                size: 256 * std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iced_video_player bind group"),
                layout: &self.bg0_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_y),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_uv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &instances,
                            offset: 0,
                            size: Some(NonZero::new(std::mem::size_of::<Uniforms>() as _).unwrap()),
                        }),
                    },
                ],
            });

            entry.insert(VideoEntry {
                texture_y,
                texture_uv,
                instances,
                bg0: bind_group,
                alive: Arc::clone(alive),
                prepare_index: AtomicUsize::new(0),
                render_index: AtomicUsize::new(0),
                _prev_dmabuf_sample: Some(sample),
                using_vulkan_export: false,
            });
        } else {
            // Replace textures and rebind for existing entry.
            let video = self.videos.get_mut(&video_id).unwrap();
            video.texture_y.destroy();
            video.texture_uv.destroy();
            video.texture_y = texture_y;
            video.texture_uv = texture_uv;

            // Store the current sample; release the previous one.
            // The previous copy command has completed by now (implicit GPU
            // serialization — this new copy creates a pipeline barrier).
            video._prev_dmabuf_sample = Some(sample);

            video.bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iced_video_player bind group"),
                layout: &self.bg0_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_y),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view_uv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &video.instances,
                            offset: 0,
                            size: Some(NonZero::new(std::mem::size_of::<Uniforms>() as _).unwrap()),
                        }),
                    },
                ],
            });
        }

        true
    }

    /// Create exportable Vulkan textures for a video and export their memory
    /// as DMA-BUF file descriptors. The returned FDs can be imported by CUDA
    /// for direct zero-copy writes (no GBM intermediary needed).
    ///
    /// This sets up the `VideoEntry` with textures that have
    /// `TEXTURE_BINDING | COPY_DST` usage and exportable backing memory.
    /// Subsequent frames rendered by CUDA writing into the exported FDs
    /// will be visible immediately — no GPU-to-GPU copy needed.
    ///
    /// Returns `Some(ExportedPlanes)` on success, `None` if the Vulkan
    /// backend or export extensions are unavailable.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn setup_vulkan_export(
        &mut self,
        device: &wgpu::Device,
        video_id: u64,
        alive: &Arc<AtomicBool>,
        (width, height): (u32, u32),
        format: VideoFormat,
    ) -> Option<ExportedPlanes> {
        use iced_wgpu::wgpu::hal::vulkan as vk_hal;

        // HAL descriptors for exportable textures — COPY_DST for potential
        // CPU fallback writes, TEXTURE_BINDING for rendering.
        let hal_y_desc = wgpu::hal::TextureDescriptor {
            label: Some("iced_video_player export Y"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.y_format(),
            usage: wgpu::TextureUses::COPY_DST | wgpu::TextureUses::RESOURCE,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };

        let hal_uv_desc = wgpu::hal::TextureDescriptor {
            label: Some("iced_video_player export UV"),
            size: wgpu::Extent3d {
                width: width / 2,
                height: height / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.uv_format(),
            usage: wgpu::TextureUses::COPY_DST | wgpu::TextureUses::RESOURCE,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };

        // wgpu public descriptors for create_texture_from_hal.
        let gpu_y_desc = wgpu::TextureDescriptor {
            label: Some("iced_video_player export Y"),
            size: hal_y_desc.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.y_format(),
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        let gpu_uv_desc = wgpu::TextureDescriptor {
            label: Some("iced_video_player export UV"),
            size: hal_uv_desc.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: format.uv_format(),
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        };

        // Create and export via Vulkan HAL.
        let (texture_y, texture_uv, exported) = unsafe {
            let hal_device_guard = match device.as_hal::<vk_hal::Api>() {
                Some(guard) => guard,
                None => {
                    warn!("Vulkan export unavailable: not a Vulkan backend");
                    return None;
                }
            };

            let hal_tex_y = match hal_device_guard.create_exportable_texture(&hal_y_desc) {
                Ok(t) => t,
                Err(e) => {
                    warn!(error = ?e, "Failed to create exportable Y texture");
                    return None;
                }
            };

            let y_export = match hal_device_guard.export_texture_memory_fd(&hal_tex_y) {
                Ok(e) => e,
                Err(e) => {
                    warn!(error = ?e, "Failed to export Y texture FD");
                    return None;
                }
            };

            let hal_tex_uv = match hal_device_guard.create_exportable_texture(&hal_uv_desc) {
                Ok(t) => t,
                Err(e) => {
                    warn!(error = ?e, "Failed to create exportable UV texture");
                    // Close the already-exported Y fd.
                    libc::close(y_export.fd);
                    return None;
                }
            };

            let uv_export = match hal_device_guard.export_texture_memory_fd(&hal_tex_uv) {
                Ok(e) => e,
                Err(e) => {
                    warn!(error = ?e, "Failed to export UV texture FD");
                    libc::close(y_export.fd);
                    return None;
                }
            };

            let texture_y = device.create_texture_from_hal::<vk_hal::Api>(hal_tex_y, &gpu_y_desc);
            let texture_uv =
                device.create_texture_from_hal::<vk_hal::Api>(hal_tex_uv, &gpu_uv_desc);

            let exported = ExportedPlanes {
                y_fd: y_export.fd,
                y_size: y_export.size,
                y_stride: y_export.row_pitch,
                uv_fd: uv_export.fd,
                uv_size: uv_export.size,
                uv_stride: uv_export.row_pitch,
            };

            (texture_y, texture_uv, exported)
        };

        info!(
            y_fd = exported.y_fd,
            y_size = exported.y_size,
            y_stride = exported.y_stride,
            uv_fd = exported.uv_fd,
            uv_size = exported.uv_size,
            uv_stride = exported.uv_stride,
            "Created Vulkan-exported textures for CUDA interop"
        );

        let view_y = texture_y.create_view(&wgpu::TextureViewDescriptor::default());
        let view_uv = texture_uv.create_view(&wgpu::TextureViewDescriptor::default());

        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("iced_video_player uniform buffer"),
            size: 256 * std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iced_video_player bind group"),
            layout: &self.bg0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_y),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_uv),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &instances,
                        offset: 0,
                        size: Some(NonZero::new(std::mem::size_of::<Uniforms>() as _).unwrap()),
                    }),
                },
            ],
        });

        // Remove any existing entry for this video.
        if let Some(old) = self.videos.remove(&video_id) {
            old.texture_y.destroy();
            old.texture_uv.destroy();
            old.instances.destroy();
        }

        self.videos.insert(
            video_id,
            VideoEntry {
                texture_y,
                texture_uv,
                instances,
                bg0: bind_group,
                alive: Arc::clone(alive),
                prepare_index: AtomicUsize::new(0),
                render_index: AtomicUsize::new(0),
                _prev_dmabuf_sample: None,
                using_vulkan_export: true,
            },
        );

        Some(exported)
    }

    fn prepare(&mut self, queue: &wgpu::Queue, video_id: u64, bounds: &iced::Rectangle) {
        if let Some(video) = self.videos.get_mut(&video_id) {
            let uniforms = Uniforms {
                rect: [
                    bounds.x,
                    bounds.y,
                    bounds.x + bounds.width,
                    bounds.y + bounds.height,
                ],
                _pad: [0; 240],
            };
            queue.write_buffer(
                &video.instances,
                (video.prepare_index.load(Ordering::Relaxed) * std::mem::size_of::<Uniforms>())
                    as u64,
                unsafe {
                    std::slice::from_raw_parts(
                        &uniforms as *const _ as *const u8,
                        std::mem::size_of::<Uniforms>(),
                    )
                },
            );
            video.prepare_index.fetch_add(1, Ordering::Relaxed);
            video.render_index.store(0, Ordering::Relaxed);
        }
    }

    fn draw(
        &self,
        target: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        clip: &iced::Rectangle<u32>,
        video_id: u64,
    ) {
        if let Some(video) = self.videos.get(&video_id) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iced_video_player render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(
                0,
                &video.bg0,
                &[
                    (video.render_index.load(Ordering::Relaxed) * std::mem::size_of::<Uniforms>())
                        as u32,
                ],
            );
            pass.set_scissor_rect(clip.x as _, clip.y as _, clip.width as _, clip.height as _);
            pass.draw(0..6, 0..1);

            video.prepare_index.store(0, Ordering::Relaxed);
            video.render_index.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VideoPrimitive {
    video_id: u64,
    alive: Arc<AtomicBool>,
    frame: Arc<Mutex<Frame>>,
    size: (u32, u32),
    format: VideoFormat,
    upload_frame: bool,
    /// Reference to cudadmabufupload element for Vulkan export setup.
    cuda_upload: Option<gst::Element>,
}

impl VideoPrimitive {
    pub fn new(
        video_id: u64,
        alive: Arc<AtomicBool>,
        frame: Arc<Mutex<Frame>>,
        size: (u32, u32),
        format: VideoFormat,
        upload_frame: bool,
        cuda_upload: Option<gst::Element>,
    ) -> Self {
        VideoPrimitive {
            video_id,
            alive,
            frame,
            size,
            format,
            upload_frame,
            cuda_upload,
        }
    }
}

impl Primitive for VideoPrimitive {
    type Pipeline = VideoPipeline;

    fn prepare(
        &self,
        pipeline: &mut VideoPipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &iced::Rectangle,
        viewport: &iced_wgpu::graphics::Viewport,
    ) {
        if self.upload_frame {
            // If Vulkan export is active for this video, CUDA already wrote
            // into our textures — no import or copy needed.
            let vulkan_export_active = pipeline
                .videos
                .get(&self.video_id)
                .is_some_and(|v| v.using_vulkan_export);

            if !vulkan_export_active {
                let frame_guard = self.frame.lock().expect("lock frame mutex");

                // On first DMA-BUF frame, try to set up Vulkan export for true
                // zero-copy (CUDA writes directly into Vulkan-owned textures).
                let is_dmabuf = frame_guard.is_dmabuf();
                if let (true, Some(element)) = (is_dmabuf, self.cuda_upload.as_ref())
                    && let Some(exported) = pipeline.setup_vulkan_export(
                        device,
                        self.video_id,
                        &self.alive,
                        self.size,
                        self.format,
                    )
                {
                    let is_p010 = self.format == VideoFormat::P010;

                    // Initialize external FD pool on the GStreamer element
                    let init_ok: bool = element.emit_by_name(
                        "init-external-pool",
                        &[&(self.size.0), &(self.size.1), &is_p010],
                    );

                    if init_ok {
                        // Add the exported buffer pair
                        let add_ok: bool = element.emit_by_name(
                            "add-external-buffer",
                            &[
                                &(exported.y_fd),
                                &(exported.y_size),
                                &(exported.y_stride as u32),
                                &(exported.uv_fd),
                                &(exported.uv_size),
                                &(exported.uv_stride as u32),
                            ],
                        );

                        if add_ok {
                            info!(
                                "Vulkan export zero-copy active (CUDA writes directly into Vulkan textures)"
                            );
                            // The video entry was created by setup_vulkan_export with
                            // using_vulkan_export = true. Subsequent frames will skip upload.
                            drop(frame_guard);
                            pipeline.prepare(
                                queue,
                                self.video_id,
                                &(*bounds
                                    * iced::Transformation::orthographic(
                                        viewport.logical_size().width as _,
                                        viewport.logical_size().height as _,
                                    )),
                            );
                            return;
                        }
                        warn!("Failed to add external buffer to GStreamer element");
                    } else {
                        warn!("Failed to init external pool on GStreamer element");
                    }
                    // Export setup failed — fall through to DMA-BUF import path.
                    // Remove the export VideoEntry so we don't render garbage.
                    if let Some(old) = pipeline.videos.remove(&self.video_id) {
                        old.texture_y.destroy();
                        old.texture_uv.destroy();
                        old.instances.destroy();
                    }
                }

                // Try DMA-BUF zero-copy path (import + copy).
                let mut used_dmabuf = false;
                if is_dmabuf && let Some(planes) = frame_guard.dmabuf_fds() {
                    trace!("DMA-BUF frame detected, importing");
                    let sample = frame_guard.sample();
                    used_dmabuf = pipeline.upload_dmabuf(
                        device,
                        queue,
                        self.video_id,
                        &self.alive,
                        self.size,
                        self.format,
                        planes,
                        sample,
                    );
                    if used_dmabuf {
                        static LOGGED_DMABUF: std::sync::atomic::AtomicBool =
                            std::sync::atomic::AtomicBool::new(false);
                        if !LOGGED_DMABUF.swap(true, Ordering::Relaxed) {
                            info!("DMA-BUF zero-copy rendering active");
                        }
                        trace!("DMA-BUF zero-copy import succeeded");
                    }
                }

                // Fallback: CPU copy via write_texture.
                if !used_dmabuf {
                    static LOGGED_CPU: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !LOGGED_CPU.swap(true, Ordering::Relaxed) {
                        info!(
                            memory_info = %frame_guard.debug_memory_info(),
                            "Using CPU copy path"
                        );
                    }
                    let stride = frame_guard.stride();
                    if let Some(readable) = frame_guard.readable() {
                        pipeline.upload(
                            device,
                            queue,
                            self.video_id,
                            &self.alive,
                            self.size,
                            self.format,
                            readable.as_slice(),
                            stride,
                        );
                    }
                }
            }
        }

        pipeline.prepare(
            queue,
            self.video_id,
            &(*bounds
                * iced::Transformation::orthographic(
                    viewport.logical_size().width as _,
                    viewport.logical_size().height as _,
                )),
        );
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &iced::Rectangle<u32>,
    ) {
        pipeline.draw(target, encoder, clip_bounds, self.video_id);
    }
}
