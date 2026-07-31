#!/usr/bin/env bash
#
# Unpack the raw DriverGaze360 release into the layout the data loader expects.
#
#   <root>/C011/001/001/            <root>/C011/001/001/
#       rgb.mp4                         rgb/000001.jpg ...
#       saliency.mp4         ->         saliency/000001.jpg ...
#       dt.mp4                          DT/000001.jpg ...
#       is.tar                          IS/000001.png ...
#       sim_gaze_df.csv                 sim_gaze_df.csv
#
# Usage: scripts/prepare_dataset.sh <dataset-root> [jobs]

set -uo pipefail

ROOT=${1:-./drivergaze360_dataset}
JOBS=${2:-$(nproc)}

# One recording. Each modality is unpacked into a .partial folder first so an
# interrupted run can just be re-run: finished folders are skipped, half-written
# ones are thrown away.
unpack() {
    rec=$1

    for pair in rgb.mp4:rgb saliency.mp4:saliency dt.mp4:DT; do
        src=$rec/${pair%%:*} out=$rec/${pair##*:}
        [[ -d $out || ! -f $src ]] && continue
        rm -rf "$out.partial" && mkdir -p "$out.partial"
        # Frame ids in sim_gaze_df.csv are 1-based, so start numbering at 1.
        ffmpeg -nostdin -loglevel error -y -i "$src" \
            -q:v 1 -qmin 1 -start_number 1 "$out.partial/%06d.jpg" </dev/null \
            && mv "$out.partial" "$out" \
            || { echo "FAILED $src" >&2; rm -rf "$out.partial"; }
    done

    if [[ ! -d $rec/IS && -f $rec/is.tar ]]; then
        rm -rf "$rec/IS.partial" && mkdir -p "$rec/IS.partial"
        if tar -xf "$rec/is.tar" -C "$rec/IS.partial"; then
            # flatten in case the archive wraps the frames in a folder
            find "$rec/IS.partial" -mindepth 2 -name '*.png' \
                -exec mv -t "$rec/IS.partial" -- {} +
            find "$rec/IS.partial" -mindepth 1 -type d -empty -delete
            mv "$rec/IS.partial" "$rec/IS"
        else
            echo "FAILED $rec/is.tar" >&2
            rm -rf "$rec/IS.partial"
        fi
    fi

    echo "done $rec"
}

# xargs re-enters the script to get parallelism across recordings
[[ ${1:-} == --unpack ]] && { unpack "$2"; exit; }

find "$ROOT" -type f -name sim_gaze_df.csv -printf '%h\0' \
    | sort -z \
    | xargs -0 -r -n1 -P "$JOBS" "$(readlink -f "$0")" --unpack
