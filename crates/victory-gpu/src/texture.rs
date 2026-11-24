//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    buffer: ID3D12Resource,
    /// Its bytes.
    #[cfg(feature = "diagnostics")]
    size: u64,
    data: *mut u8,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    last: Mutex<Last>,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after,
/// with room to read the memory into.
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    read: Vec<u8>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    set.get_or_insert_with(Default::default)
        .insert(resource as usize);
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // The staging buffer lives as long as `staging`, which the deferred
    // refresh below holds until the list is reset.
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (staging.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&staging.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging);
}

/// Whether placed copies are brought up to date from the resource that
/// writes their memory; on unless `VICTORY_PLACED_SYNC=0` (to measure it).
fn placed_sync_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_PLACED_SYNC").as_deref() != Some(std::ffi::OsStr::new("0"))
    })
}

/// The staged copy of one placed copy's contents from the resource that
/// writes its memory. The two formats are of different families (the writer
/// writes 32-bit words, the reader's is the console's packed one), which the
/// host refuses to copy between, so the texels go through a buffer that
/// names the writer's format at one end and the placed copy's at the other.
struct PlaceSync {
    buffer: ID3D12Resource,
    /// Per subresource, with the writer's format (texture to buffer).
    from: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    /// The same, with the placed copy's format (buffer to texture).
    to: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
}

static PLACE_SYNCS: Mutex<Option<HashMap<(usize, usize), Arc<PlaceSync>>>> = Mutex::new(None);

/// Copies of placed copies recorded since the frame line last read it.
static SYNCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The placed copies recorded since the last call (frame line, diagnostics).
pub(crate) fn take_syncs() -> u64 {
    SYNCS.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// The staging state of the pair (writer, placed copy), made on first use.
fn place_sync(
    writer: &ID3D12Resource,
    placed: &ID3D12Resource,
    address: u64,
) -> Option<Arc<PlaceSync>> {
    use windows::core::Interface;
    let key = (writer.as_raw() as usize, placed.as_raw() as usize);
    {
        let map = PLACE_SYNCS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cell) = map.as_ref().and_then(|m| m.get(&key)) {
            return Some(cell.clone());
        }
    }
    let host = host()?;
    // SAFETY: live resources.
    let (from_desc, to_desc) = unsafe { (writer.GetDesc(), placed.GetDesc()) };
    let count = if from_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(from_desc.MipLevels)
    } else {
        u32::from(from_desc.MipLevels) * u32::from(from_desc.DepthOrArraySize)
    };
    let mut from = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut total = 0u64;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &from_desc,
            0,
            count,
            0,
            Some(from.as_mut_ptr()),
            None,
            None,
            Some(&mut total),
        );
    }
    if total == 0 {
        return None;
    }
    // The same rows both ways (the two formats are the same bytes per
    // texel); the footprint names whichever format that end's resource has.
    let to: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> = from
        .iter()
        .map(|f| {
            let mut f = *f;
            f.Footprint.Format = to_desc.Format;
            f
        })
        .collect();
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a staging buffer.
    unsafe {
        host.device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
    }
    let buffer = buffer?;
    crate::host::name(&buffer, || format!("placed copy staging {total:#x}"));
    tracing::info!(
        bytes = total,
        subresources = count,
        from = from_desc.Format.0,
        to = to_desc.Format.0,
        width = from_desc.Width,
        height = from_desc.Height,
        depth = from_desc.DepthOrArraySize,
        address = format_args!("{address:#x}"),
        "placed copy: staged between the writer's format and the placed one"
    );
    let cell = Arc::new(PlaceSync { buffer, from, to });
    let mut map = PLACE_SYNCS.lock().unwrap_or_else(|e| e.into_inner());
    map.get_or_insert_with(HashMap::new)
        .insert(key, cell.clone());
    Some(cell)
}

/// Records a barrier on the staging buffer, whose state every list leaves as
/// it found it: copy destination at the start of each copy, since no
/// resource state is tracked for it.
fn staging_barrier(list: &HostList, buffer: &ID3D12Resource, before: u32, after: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    let barrier = crate::host::transition(buffer, State(before as i32), State(after as i32));
    // SAFETY: an open list and the staging buffer, which only these copies
    // use.
    unsafe { list.list.ResourceBarrier(&[barrier]) };
}

/// Brings a placed copy of title memory up to date with the resource that
/// writes that memory (a UAV the title's shaders write), in this list and so
/// before the draw or dispatch that reads it: the placed copy is made from
/// the title bytes, which a GPU write never reaches.
///
/// Returns whether the placed copy's contents come from the writer (a copy
/// recorded in this list, or already recorded in it), so the caller can
/// leave out an upload of title memory that holds none of them.
pub(crate) fn sync_placed(list: &HostList, placed: This, view: &crate::descriptor::View) -> bool {
    if !placed_sync_on() {
        return false;
    }
    let crate::descriptor::View::Placed(false, resource, _) = view else {
        return false;
    };
    const BUFFER: u32 = 1;
    if resource[0] == BUFFER {
        return false;
    }
    let Some(placed_copy) = crate::resource::host_copy(placed) else {
        return false;
    };
    let Some((memory, _)) = crate::resource::texture_memory(placed) else {
        return false;
    };
    let Some(writer) = crate::resource::writer_at(memory as u64, placed, &placed_copy.texture)
    else {
        return false;
    };
    let Some(cell) = place_sync(&writer.texture, &placed_copy.texture, memory as u64) else {
        return false;
    };
    if !list.first_sync(Arc::as_ptr(&placed_copy) as usize) {
        return true;
    }
    SYNCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    // The writer's shaders may have written it: a barrier orders the copy
    // after them (the prelude puts it in that state when this list first
    // uses it).
    crate::list::ensure_state_host(list, &writer, COPY_SOURCE);
    crate::list::ensure_state_host(list, &placed_copy, COPY_DEST);
    for (i, from) in cell.from.iter().enumerate() {
        let source = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointers the locations never release.
            pResource: unsafe { std::mem::transmute_copy(&writer.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let staging = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&cell.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *from,
            },
        };
        // SAFETY: an open list; both locations name live resources of the
        // same size and texel width.
        #[cfg(feature = "diagnostics")]
        {
            let (from_f, to_f) = (
                cell.from[i].Footprint.Format.0,
                cell.to[i].Footprint.Format.0,
            );
            let (w, h, d) = (
                cell.from[i].Footprint.Width,
                cell.from[i].Footprint.Height,
                cell.from[i].Footprint.Depth,
            );
            list.note_read(
                || {
                    format!(
                        "placed copy {:#x} sub {i} of {} from the writer: {:#x} {w}x{h}x{d} f{from_f} to f{to_f}",
                        placed as usize,
                        cell.from.len(),
                        memory as usize,
                    )
                },
                (0, 0),
            );
        }
        unsafe {
            list.list
                .CopyTextureRegion(&staging, 0, 0, 0, &source, None)
        };
    }
    staging_barrier(list, &cell.buffer, COPY_DEST, COPY_SOURCE);
    for (i, to) in cell.to.iter().enumerate() {
        let staging = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointers the locations never release.
            pResource: unsafe { std::mem::transmute_copy(&cell.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *to,
            },
        };
        let placed_at = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&placed_copy.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: an open list; the staging buffer covers the subresource.
        #[cfg(feature = "diagnostics")]
        {
            let (from_f, to_f) = (
                cell.from[i].Footprint.Format.0,
                cell.to[i].Footprint.Format.0,
            );
            let (w, h, d) = (
                cell.to[i].Footprint.Width,
                cell.to[i].Footprint.Height,
                cell.to[i].Footprint.Depth,
            );
            list.note_read(
                || {
                    format!(
                        "placed copy {:#x} sub {i} of {} to the placed copy: {:#x} {w}x{h}x{d} f{from_f} to f{to_f}",
                        placed as usize,
                        cell.from.len(),
                        memory as usize,
                    )
                },
                (0, 0),
            );
        }
        unsafe {
            list.list
                .CopyTextureRegion(&placed_at, 0, 0, 0, &staging, None)
        };
    }
    staging_barrier(list, &cell.buffer, COPY_SOURCE, COPY_DEST);
    tracing::debug!(
        placed = format_args!("{:#x}", placed as usize),
        subresources = cell.from.len(),
        address = format_args!("{:#x}", memory as usize),
        "placed copy synced from the resource that writes its memory"
    );
    true
}

impl Staging {
    /// The title memory the texture is read from: start and end.
    pub(crate) fn range(&self) -> (usize, usize) {
        let start = self.memory as usize;
        (start, start + self.layout.size as usize)
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some((now, written)) = crate::watch::writes(start, start + size, false) {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            last.read.resize(size, 0);
            crate::host::copy_mapped(last.read.as_mut_ptr(), self.memory, size, false);
            self.detile(&last.read);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return 0;
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return size;
        }
        last.hash = Some(hash);
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8]) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear = unsafe {
                std::slice::from_raw_parts_mut(self.data.add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let (buffer, data) = host.upload_buffer(total)?;
    crate::host::name(&buffer, || {
        format!(
            "staging of {resource:?} {}x{}x{} f{}",
            d.Width, d.Height, d.DepthOrArraySize, d.Format.0
        )
    });
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let s = Arc::new(Staging {
        buffer,
        #[cfg(feature = "diagnostics")]
        size: total,
        data,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        last: Mutex::default(),
    });
    map.insert(resource as usize, s.clone());
    Some(s)
}
