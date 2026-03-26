//! DMA-BUF zero-copy import verification test.
//!
//! Imports a DMA-BUF NV12 frame's Y and UV planes into wgpu textures via the
//! Vulkan HAL, reads them back to CPU, and compares against the CPU-mapped
//! reference data from the same GStreamer buffer.
//!
//! Usage:
//!   GST_PLUGIN_PATH=/path/to/gst-cuda-dmabuf/builddir/src \
//!     cargo run --example dmabuf_verify

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_allocators::DmaBufMemory;
use gstreamer_app as gst_app;
use gstreamer_video::VideoMeta;
use iced_wgpu::wgpu;
use iced_wgpu::wgpu::hal::vulkan as vk_hal;

const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

fn align_up(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "iced_video_player=info".into()),
        )
        .init();

    // ── GStreamer: pull one DMA-BUF NV12 frame ──────────────────────────
    gst::init().unwrap();

    let path = std::fs::canonicalize(".media/test.mp4").expect(".media/test.mp4 not found");

    let pipeline_str = format!(
        "filesrc location=\"{}\" ! qtdemux ! h264parse ! nvh264dec ! \
         cudadmabufupload force-linear=true ! appsink name=vsink drop=true",
        path.display()
    );
    let pipeline = gst::parse::launch(&pipeline_str)
        .expect("Failed to create pipeline")
        .dynamic_cast::<gst::Pipeline>()
        .unwrap();

    let vsink = pipeline
        .by_name("vsink")
        .and_then(|e| e.dynamic_cast::<gst_app::AppSink>().ok())
        .expect("appsink not found — is GST_PLUGIN_PATH set?");

    pipeline
        .set_state(gst::State::Playing)
        .expect("Failed to start pipeline");

    // Pull first frame
    let sample = vsink.pull_sample().expect("Failed to pull sample");

    // Pause immediately to prevent GStreamer from overwriting the DMA-BUF buffer
    pipeline
        .set_state(gst::State::Paused)
        .expect("Failed to pause pipeline");

    let buffer = sample.buffer().unwrap();

    // Frame dimensions from caps
    let caps = sample.caps().unwrap();
    let s = caps.structure(0).unwrap();
    println!("Caps: {s}");

    // For DMA_DRM format, get dimensions from the structure
    let width: i32 = s.get("width").unwrap();
    let height: i32 = s.get("height").unwrap();
    let width = width as u32;
    let height = height as u32;
    println!("Frame: {width}x{height}");

    // DMA-BUF metadata
    let video_meta = buffer.meta::<VideoMeta>().expect("No VideoMeta on buffer");
    let strides = video_meta.stride();
    let offsets = video_meta.offset();
    let y_stride = strides[0] as u32;
    let uv_stride = strides[1] as u32;
    let y_offset = offsets[0] as u32;
    let uv_offset = offsets[1] as u32;
    println!("Y  plane: stride={y_stride}, offset={y_offset}");
    println!("UV plane: stride={uv_stride}, offset={uv_offset}");

    let mem0 = buffer.peek_memory(0);
    let dmabuf = mem0
        .downcast_memory_ref::<DmaBufMemory>()
        .expect("Memory is not DMA-BUF");
    let base_fd = dmabuf.fd();
    println!("DMA-BUF fd={base_fd}, mem_size={}", mem0.size());

    // CPU reference: map the DMA-BUF for reading
    let map = buffer.map_readable().expect("Failed to CPU-map buffer");
    let cpu_data = map.as_slice();
    println!("CPU mapped {} bytes", cpu_data.len());

    // ── wgpu: create headless Vulkan device ─────────────────────────────
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No Vulkan adapter found");

    println!("GPU: {}", adapter.get_info().name);

    let features = adapter.features();
    assert!(
        features.contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF),
        "Adapter lacks VULKAN_EXTERNAL_MEMORY_DMA_BUF"
    );

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("dmabuf_verify"),
        required_features: wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF,
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        ..Default::default()
    }))
    .expect("Failed to create device");

    // ── Test Y plane (direct readback) ─────────────────────────────────
    println!("\n=== Y Plane: Direct Import Readback ===");
    let y_result = test_plane_import(
        &device,
        &queue,
        base_fd,
        width,
        height,
        y_stride,
        y_offset,
        0, // total_size=0 → dedicated allocation
        wgpu::TextureFormat::R8Unorm,
        1, // bytes per pixel
        cpu_data,
        "Y",
    );

    // ── Test UV plane (direct readback) ─────────────────────────────────
    println!("\n=== UV Plane: Direct Import Readback ===");
    let uv_w = width / 2;
    let uv_h = height / 2;
    let uv_total_size = uv_offset as u64 + uv_stride as u64 * uv_h as u64;
    let uv_result = test_plane_import(
        &device,
        &queue,
        base_fd,
        uv_w,
        uv_h,
        uv_stride,
        uv_offset,
        uv_total_size,
        wgpu::TextureFormat::Rg8Unorm,
        2, // bytes per pixel
        cpu_data,
        "UV",
    );

    // ── Test import→copy→readback (production path) ─────────────────────
    println!("\n=== Y Plane: Import → Copy → Readback (production path) ===");
    let y_copy_result = test_plane_import_copy(
        &device,
        &queue,
        base_fd,
        width,
        height,
        y_stride,
        y_offset,
        0,
        wgpu::TextureFormat::R8Unorm,
        1,
        cpu_data,
        "Y-copy",
    );

    println!("\n=== UV Plane: Import → Copy → Readback (production path) ===");
    let uv_copy_result = test_plane_import_copy(
        &device,
        &queue,
        base_fd,
        uv_w,
        uv_h,
        uv_stride,
        uv_offset,
        uv_total_size,
        wgpu::TextureFormat::Rg8Unorm,
        2,
        cpu_data,
        "UV-copy",
    );

    // ── Summary ─────────────────────────────────────────────────────────
    pipeline.set_state(gst::State::Null).unwrap();

    let sep = "=".repeat(60);
    println!("\n{sep}");
    println!("  Direct import:");
    println!(
        "    Y plane:  {} ({} mismatches, max_diff={})",
        if y_result.pass { "PASS" } else { "FAIL" },
        y_result.mismatches,
        y_result.max_diff
    );
    println!(
        "    UV plane: {} ({} mismatches, max_diff={})",
        if uv_result.pass { "PASS" } else { "FAIL" },
        uv_result.mismatches,
        uv_result.max_diff
    );
    println!("  Import → Copy (production path):");
    println!(
        "    Y plane:  {} ({} mismatches, max_diff={})",
        if y_copy_result.pass { "PASS" } else { "FAIL" },
        y_copy_result.mismatches,
        y_copy_result.max_diff
    );
    println!(
        "    UV plane: {} ({} mismatches, max_diff={})",
        if uv_copy_result.pass { "PASS" } else { "FAIL" },
        uv_copy_result.mismatches,
        uv_copy_result.max_diff
    );
    println!("{sep}");

    let all_pass = y_result.pass && uv_result.pass && y_copy_result.pass && uv_copy_result.pass;
    if all_pass {
        println!("\nPASS: DMA-BUF import produces pixel-identical data.");
    } else {
        println!("\nFAIL: DMA-BUF import data differs from CPU reference.");
    }

    std::process::exit(if all_pass { 0 } else { 1 });
}

struct PlaneResult {
    pass: bool,
    mismatches: u64,
    max_diff: u8,
}

#[allow(clippy::too_many_arguments)]
fn test_plane_import(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_fd: i32,
    tex_width: u32,
    tex_height: u32,
    stride: u32,
    offset: u32,
    total_size: u64,
    format: wgpu::TextureFormat,
    bpp: u32,
    cpu_data: &[u8],
    label: &str,
) -> PlaneResult {
    let fd = unsafe { libc::dup(base_fd) };
    assert!(fd >= 0, "dup() failed for {label}");

    // wgpu public texture descriptor (needs COPY_SRC for readback)
    let wgpu_desc = wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: tex_width,
            height: tex_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };

    // HAL texture descriptor
    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some(label),
        size: wgpu_desc.size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    let plane_info = vk_hal::DmaBufPlaneInfo {
        drm_modifier: 0,
        stride,
        offset,
        total_size,
    };

    println!(
        "  Importing {label}: {tex_width}x{tex_height} {format:?}, stride={stride}, offset={offset}"
    );

    let texture = unsafe {
        let guard = device.as_hal::<vk_hal::Api>().expect("Not a Vulkan device");
        let hal_tex = guard
            .texture_from_dmabuf_fd(fd, &hal_desc, &plane_info)
            .unwrap_or_else(|e| panic!("{label} DMA-BUF import failed: {e:?}"));
        device.create_texture_from_hal::<vk_hal::Api>(hal_tex, &wgpu_desc)
    };

    println!("  Import succeeded. Reading back...");

    // Readback: copy texture → staging buffer
    let row_bytes = tex_width * bpp;
    let aligned_bpr = align_up(row_bytes, ALIGN);
    let buf_size = aligned_bpr as u64 * tex_height as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} staging")),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bpr),
                rows_per_image: Some(tex_height),
            },
        },
        wgpu::Extent3d {
            width: tex_width,
            height: tex_height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    // Map and read
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let gpu_data = slice.get_mapped_range();

    // Compare pixel by pixel
    let mut mismatches = 0u64;
    let mut max_diff = 0u8;
    let total_pixels = tex_width as u64 * tex_height as u64 * bpp as u64;

    for row in 0..tex_height as usize {
        let cpu_row_start = offset as usize + row * stride as usize;
        let gpu_row_start = row * aligned_bpr as usize;
        for col in 0..row_bytes as usize {
            let cpu_val = cpu_data[cpu_row_start + col];
            let gpu_val = gpu_data[gpu_row_start + col];
            if cpu_val != gpu_val {
                mismatches += 1;
                let diff = cpu_val.abs_diff(gpu_val);
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
    }

    let pct = mismatches as f64 / total_pixels as f64 * 100.0;
    println!(
        "  {label}: {mismatches}/{total_pixels} byte mismatches ({pct:.2}%), max_diff={max_diff}"
    );

    // Diagnostics: dump first 2 rows if there are mismatches
    if mismatches > 0 {
        println!("  Diagnostic: first 2 rows of {label}");
        for r in 0..2 {
            let cpu_start = offset as usize + r * stride as usize;
            let gpu_start = r * aligned_bpr as usize;
            let cpu_row: Vec<u8> = (0..16.min(row_bytes as usize))
                .map(|i| cpu_data[cpu_start + i])
                .collect();
            let gpu_row: Vec<u8> = (0..16.min(row_bytes as usize))
                .map(|i| gpu_data[gpu_start + i])
                .collect();
            println!("    row{r} CPU[0..16]: {cpu_row:?}");
            println!("    row{r} GPU[0..16]: {gpu_row:?}");
        }

        // Check if it looks like a stride offset issue
        // If GPU row N matches CPU row N starting at a shifted offset, it's stride mismatch
        let _cpu_row1_start = offset as usize + stride as usize;
        let gpu_row1_start = aligned_bpr as usize;
        let shift_check_bytes = 32.min(row_bytes as usize);

        // Check several possible strides
        for test_stride in [row_bytes, row_bytes + 64, row_bytes + 128, stride] {
            let test_stride = test_stride as usize;
            let mut matches = 0;
            for i in 0..shift_check_bytes {
                if offset as usize + test_stride + i < cpu_data.len()
                    && cpu_data[offset as usize + test_stride + i] == gpu_data[gpu_row1_start + i]
                {
                    matches += 1;
                }
            }
            println!(
                "    GPU row1 vs CPU @stride={test_stride}: {matches}/{shift_check_bytes} matches"
            );
        }
    }

    drop(gpu_data);
    staging.unmap();

    PlaneResult {
        pass: mismatches == 0,
        mismatches,
        max_diff,
    }
}

/// Tests the production path: import DMA-BUF → copy to GPU texture → readback.
#[allow(clippy::too_many_arguments)]
fn test_plane_import_copy(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    base_fd: i32,
    tex_width: u32,
    tex_height: u32,
    stride: u32,
    offset: u32,
    total_size: u64,
    format: wgpu::TextureFormat,
    bpp: u32,
    cpu_data: &[u8],
    label: &str,
) -> PlaneResult {
    let fd = unsafe { libc::dup(base_fd) };
    assert!(fd >= 0, "dup() failed for {label}");

    // Import descriptor (COPY_SRC only — matches production code)
    let import_desc = wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: tex_width,
            height: tex_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };

    let hal_desc = wgpu::hal::TextureDescriptor {
        label: Some(label),
        size: import_desc.size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUses::COPY_SRC,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: vec![],
    };

    let plane_info = vk_hal::DmaBufPlaneInfo {
        drm_modifier: 0,
        stride,
        offset,
        total_size,
    };

    println!(
        "  Importing {label}: {tex_width}x{tex_height} {format:?}, stride={stride}, offset={offset}"
    );

    let imported = unsafe {
        let guard = device.as_hal::<vk_hal::Api>().expect("Not a Vulkan device");
        let hal_tex = guard
            .texture_from_dmabuf_fd(fd, &hal_desc, &plane_info)
            .unwrap_or_else(|e| panic!("{label} DMA-BUF import failed: {e:?}"));
        device.create_texture_from_hal::<vk_hal::Api>(hal_tex, &import_desc)
    };

    // GPU-local texture (matches production: COPY_DST | TEXTURE_BINDING)
    let gpu_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(&format!("{label} gpu")),
        size: import_desc.size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // Copy imported → GPU (production path)
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_texture(
        imported.as_image_copy(),
        gpu_tex.as_image_copy(),
        import_desc.size,
    );
    queue.submit(Some(encoder.finish()));
    drop(imported);

    println!("  Copy to GPU succeeded. Reading back from GPU texture...");

    // Readback from GPU texture
    let row_bytes = tex_width * bpp;
    let aligned_bpr = align_up(row_bytes, ALIGN);
    let buf_size = aligned_bpr as u64 * tex_height as u64;

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} staging")),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        gpu_tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(aligned_bpr),
                rows_per_image: Some(tex_height),
            },
        },
        wgpu::Extent3d {
            width: tex_width,
            height: tex_height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let gpu_data = slice.get_mapped_range();

    // Compare
    let mut mismatches = 0u64;
    let mut max_diff = 0u8;
    let total_pixels = tex_width as u64 * tex_height as u64 * bpp as u64;

    for row in 0..tex_height as usize {
        let cpu_row_start = offset as usize + row * stride as usize;
        let gpu_row_start = row * aligned_bpr as usize;
        for col in 0..row_bytes as usize {
            let cpu_val = cpu_data[cpu_row_start + col];
            let gpu_val = gpu_data[gpu_row_start + col];
            if cpu_val != gpu_val {
                mismatches += 1;
                let diff = cpu_val.abs_diff(gpu_val);
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
    }

    let pct = mismatches as f64 / total_pixels as f64 * 100.0;
    println!(
        "  {label}: {mismatches}/{total_pixels} byte mismatches ({pct:.2}%), max_diff={max_diff}"
    );

    drop(gpu_data);
    staging.unmap();

    PlaneResult {
        pass: mismatches == 0,
        mismatches,
        max_diff,
    }
}
