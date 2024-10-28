using Downloads, ProgressMeter

function download_with_progress(url, path; n_parts=1_000)
    p = Progress(n_parts; dt=0.5, barlen=64, desc="⬇️  $(url)")
    path = Downloads.download(url, path, progress=(total, now) -> (total > 0 && total != now) ? update!(p, (now*n_parts)÷total) : nothing)
    finish!(p)
    return path
end
