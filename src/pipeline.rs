use crate::video::{DmaBufPlanes, Frame};
use gstreamer as gst;
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
    fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        video_id: u64,
        alive: &Arc<AtomicBool>,
        (width, height): (u32, u32),
        frame: &[u8],
        stride: Option<u32>,
    ) {
        // Use stride from GStreamer's VideoMeta if available, otherwise assume stride == width
        let stride = stride.unwrap_or(width);
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
                format: wgpu::TextureFormat::R8Unorm,
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
                format: wgpu::TextureFormat::Rg8Unorm,
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
    fn upload_dmabuf(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        video_id: u64,
        alive: &Arc<AtomicBool>,
        (width, height): (u32, u32),
        planes: DmaBufPlanes,
        sample: gst::Sample,
    ) -> bool {
        use iced_wgpu::wgpu::hal::vulkan as vk_hal;

        let y_fd = planes.y_fd;
        let uv_fd = planes.uv_fd;

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
            format: wgpu::TextureFormat::R8Unorm,
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
            format: wgpu::TextureFormat::Rg8Unorm,
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
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        };

        let gpu_uv_desc = wgpu::TextureDescriptor {
            label: Some("iced_video_player UV"),
            size: import_uv_desc.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
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
                    libc::close(y_fd);
                    libc::close(uv_fd);
                    return false;
                }
            };

            let hal_tex_y =
                match hal_device_guard.texture_from_dmabuf_fd(y_fd, &hal_y_desc, &y_plane_info) {
                    Ok(t) => t,
                    Err(e) => {
                        warn!(error = ?e, "DMA-BUF Y plane import failed");
                        // y_fd was consumed by the failed attempt; only close uv_fd
                        libc::close(uv_fd);
                        return false;
                    }
                };

            let hal_tex_uv = match hal_device_guard.texture_from_dmabuf_fd(
                uv_fd,
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
    upload_frame: bool,
}

impl VideoPrimitive {
    pub fn new(
        video_id: u64,
        alive: Arc<AtomicBool>,
        frame: Arc<Mutex<Frame>>,
        size: (u32, u32),
        upload_frame: bool,
    ) -> Self {
        VideoPrimitive {
            video_id,
            alive,
            frame,
            size,
            upload_frame,
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
            let frame_guard = self.frame.lock().expect("lock frame mutex");

            // Try DMA-BUF zero-copy path first.
            let mut used_dmabuf = false;
            let is_dmabuf = frame_guard.is_dmabuf();
            if is_dmabuf && let Some(planes) = frame_guard.dmabuf_fds() {
                trace!("DMA-BUF frame detected, importing");
                let sample = frame_guard.sample();
                used_dmabuf = pipeline.upload_dmabuf(
                    device,
                    queue,
                    self.video_id,
                    &self.alive,
                    self.size,
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
                        readable.as_slice(),
                        stride,
                    );
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
