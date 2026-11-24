//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Of a title copy: waiting for earlier submissions, flushing pending
    /// refreshes, checking the destination is mapped, opening its guards,
    /// and the reads that met an unmapped page.
    CopySettle,
    CopyFlush,
    CopyCommitted,
    CopyGuard,
    CopyFallback,
    /// Of copying shader writes back: bytes compared with the snapshot, and
    /// the time reading title memory, merging the bytes, writing them, and
    /// opening the guards before a write.
    BackCompared,
    BackRead,
    BackMerge,
    BackWrite,
    BackGuard,
    /// Writes attempted and failed: a failed one halves its range.
    WriteCalls,
    WriteFails,
    /// Time in the title's draw, dispatch and copy entry points, and how
    /// many calls.
    EntryUs,
    EntryCalls,
    /// Of a title call: preparing it (bindings, vertex buffers, targets) and
    /// the driver's draw call itself; of preparing, making its tables.
    DrawPrepUs,
    DrawTablesUs,
    DrawCallUs,
    Draws,
    /// Table lookups that found the table already made, that had to make it,
    /// and the descriptor slots made with it.
    TableHits,
    TableMisses,
    TableEntries,
    /// Every command-list call: its time and count.
    ListCallUs,
    ListCalls,
}
const COUNTERS: usize = 46;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Whether shader writes go back into title memory with a plain copy (timing
/// only, `VICTORY_WRITE_MODE=plain`).
pub(crate) fn write_plain() -> bool {
    static PLAIN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PLAIN.get_or_init(|| std::env::var("VICTORY_WRITE_MODE").as_deref() == Ok("plain"))
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks; {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
    );
    tracing::info!(
        "of a title copy: {:.1} ms waiting for earlier submissions, {:.1} ms flushing refreshes, \
         {:.1} ms checking the destination, {:.1} ms opening its guards, {:.1} ms reading around \
         unmapped pages",
        ms(c[C::CopySettle as usize]),
        ms(c[C::CopyFlush as usize]),
        ms(c[C::CopyCommitted as usize]),
        ms(c[C::CopyGuard as usize]),
        ms(c[C::CopyFallback as usize]),
    );
    tracing::info!(
        "of copying shader writes back: {:.1} MB compared, {:.1} ms reading title memory, \
         {:.1} ms merging, {:.1} ms writing ({:.1} ms opening guards; {:.1} writes, {:.1} failed)",
        mb(c[C::BackCompared as usize]),
        ms(c[C::BackRead as usize]),
        ms(c[C::BackMerge as usize]),
        ms(c[C::BackWrite as usize]),
        ms(c[C::BackGuard as usize]),
        c[C::WriteCalls as usize],
        c[C::WriteFails as usize],
    );
    tracing::info!(
        "of the title's calls: {:.2} ms in {} draws, dispatches and copies",
        ms(c[C::EntryUs as usize]),
        c[C::EntryCalls as usize],
    );
    tracing::info!(
        "of a draw: {:.1} ms preparing (of which {:.1} ms tables), {:.1} ms in the driver's call, \
         {:.0} draws",
        ms(c[C::DrawPrepUs as usize]),
        ms(c[C::DrawTablesUs as usize]),
        ms(c[C::DrawCallUs as usize]),
        c[C::Draws as usize],
    );
    tracing::info!(
        "tables: {:.0} hits, {:.0} misses, {:.0} descriptor slots made",
        c[C::TableHits as usize],
        c[C::TableMisses as usize],
        c[C::TableEntries as usize],
    );
    tracing::info!(
        "command lists: {:.1} ms in {} calls",
        ms(c[C::ListCallUs as usize]),
        c[C::ListCalls as usize],
    );
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}
