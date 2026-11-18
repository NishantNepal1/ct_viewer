# CT segmentation viewer

A small local web app that loads a DICOM CT series, uploads it to the Nucleo segmentation API, overlays the returned masks next to a reference segmentation, and shows the Dice score.

Python (FastAPI) backend + one static page (vanilla JS, `<canvas>`). The backend keeps the API token out of the browser, avoids CORS, and does the DICOM/NIfTI/Dice work in numpy.

## Run

```bash
cp .env.example .env      # then put your token in .env
./run.sh                  # or: make run
```

Open http://localhost:8000. `run.sh` creates `.venv` with Python ≥ 3.11 if it's missing, installs `requirements.txt`, and starts uvicorn on 127.0.0.1 only.

Tests: `make test` (or `./run.sh test`). They run offline: the API is stubbed and all data is synthetic.

## Configuration

Set these in `.env` or as environment variables:

| Variable                  | Default                                           | Purpose                                                                                            |
| ------------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `NUCLEO_API_TOKEN`      | –                                                | Bearer token. If it's missing, the app still runs and says "API token not configured".             |
| `NUCLEO_API_BASE_URL`   | `https://bnj2vb4zpt.us-east-1.awsapprunner.com` | API base URL (a trailing slash is stripped).                                                       |
| `REFERENCE_MASK_PATH`   | `data/case-001-reference-lesions.nii.gz`        | Reference lesion mask. If it's missing, the app still works without Dice or the reference overlay. |
| `LESION_STRUCTURE_NAME` | `lesions`                                       | Which returned structure is compared with the reference.                                           |

All limits (upload size, zip caps, timeouts, retries, concurrency, memory) are named constants in `app/config.py`.

## Data

Put the files in `data/` (it's gitignored):

```
data/case-001-dicom.zip
data/case-001-reference-lesions.nii.gz
```

Pick the zip in the browser. The reference is loaded from `REFERENCE_MASK_PATH` at startup.

## How it works

1. **Load.** The browser checks the size (≤ 150 MiB) and the `PK\x03\x04` magic, then POSTs the raw bytes to `/api/series`. The server parses them in a thread pool and keeps the zip and the CT in memory under a `series_id`. The browser fetches the CT once as Int16 and does all windowing on the client.
2. **Run.** `POST /api/series/{id}/jobs` returns a `job_id` at once. A background task uploads the zip, downloads each returned structure, validates it, maps it onto the CT grid and computes Dice. The page polls `/api/jobs` every 500 ms.
3. **Show.** Masks go to the browser as bit-packed uint8 on the CT display grid (`np.packbits`, about 3 MB instead of 26 MB). Model structures are drawn as semi-transparent fills, one colour each, with toggles built from the response. The reference is a green outline with its own toggle.

## Geometry

- **Slice order.** Slices are sorted by the projection of `ImagePositionPatient` onto the slice normal (row cosine × column cosine). Filenames and `InstanceNumber` are ignored. Duplicate positions are refused. Uneven spacing gives a warning above 1 % and is refused above 10 % (likely a missing slice).
- **CT affine.** Built from IOP, `PixelSpacing` (`[row, column]` spacing), the first slice's IPP and the mean slice step. Index order is `(i, j, k) = (column, row, slice)`, the same as NIfTI. It's converted to RAS with `diag(-1, -1, 1, 1)` so it can be compared with NIfTI affines.
- **Display layout.** The display array is `(k, j, i)`, so `volume[k]` is exactly the DICOM pixel array of slice `k`. It's shown as stored (radiological view) with L/R/A/P labels worked out from IOP.
- **Mapping a mask onto the CT.** `M = inv(A_ct) @ A_mask` maps mask voxels to CT voxels. If `M` is a signed permutation with integer translation (within 1e-3 voxel) and the mask fits inside the CT, it's applied exactly with transpose, flip and slicing. Otherwise the overlay is refused with a message; nothing is resampled silently. There's no `as_closest_canonical` and no `reshape`.
- **Real case.** For case-001, the DICOM affine equals the reference/mask affine `diag(-0.709, -0.709, 1)`. The NIfTI origin `(170.18, 277.27, 359.9)` in RAS is the first sorted slice's IPP `(-170.18, -277.27, 359.9)` in LPS, so `M` is the identity.

## Dice

`DSC = 2|A∩B| / (|A| + |B|)` on boolean masks, after binarising with `> 0.5`. Non-0/1 values are reported, not silently accepted.

- It's computed only if `same_grid(model, reference)` passes on the original NIfTI grids (same shape, and affines equal within 1e-3 mm + 1e-4 relative). Otherwise the app shows why the grids differ instead of a number.
- If both masks are empty, Dice is undefined and the app says so (it doesn't show 1.0 or divide by zero).
- It's computed only when the API says the scan is `case-001`, which is the case the reference belongs to.
- The structure compared is the one named `LESION_STRUCTURE_NAME`. If the model didn't return it, the app says so and offers a dropdown of the structures it did return.
- Voxel counts and volumes (mL, from the affine determinant) are shown for both masks.

## Concurrency and retries

- **Jobs.** Each upload is its own job with a random id. Results are stored on the job, never keyed by mask URL or structure name, because both uploads of the same case return the same mask URL. By default the page shows the most recently *started* finished job, so a slower, older job that finishes later doesn't replace it. You can click any job to view it.
- **Upload slots.** An `asyncio.Semaphore(2)` matches the token's 2-concurrent-upload limit. Extra jobs show "Waiting for a free upload slot" instead of triggering 429s.
- **429 / 503.** These are retried up to 5 attempts in total. The wait is at least `Retry-After` (integer seconds or an HTTP-date), falling back to `details.retry_after_s`. Full-jitter exponential backoff is also applied, and with no hint the backoff alone is used. If the server asks for more than 60 s, the app stops and tells the user rather than retrying early. The job status shows "Server busy, retrying in N s".
- **Network errors and 502/504.** These get up to 3 attempts in total.
- **Never retried:** 400, 401, 404, 405, 408, 413, 415, 422, and a 429 with `limit = failed_auth_per_ip` (bad token; the message says to check it).
- **Mask URLs.** These are joined onto the base URL as strings, so a base path prefix is kept. An absolute URL on another host is refused and no request is sent, so the token can't leak. Redirects aren't followed.

## Failure cases handled (each has a test)

- **Bad local files:** empty file, not a zip, truncated or corrupt zip, a zip with no DICOM, non-DICOM files mixed in, two series, more than 5,000 entries, over the per-member or total expanded size cap, over 150 MiB, a truncated DICOM, missing tags, duplicate or missing slices. Each gives a plain-language error, and the server keeps serving.
- **API errors:** every error code and reason in the brief maps to a plain message (`app/errors.py`). The raw code and reason are under "Technical details", never a stack trace. Bodies that are non-JSON, empty or partial (for example gateway HTML) are handled.
- **Bad 200 responses:** non-JSON, missing or ill-typed `case_id`/`structures`, or an empty structure list each give a friendly error.
- **Masks:** non-gzip data, corrupt or truncated gzip, decompression bombs (bounded streaming gunzip), non-NIfTI bytes, non-3D masks, NaNs, non-binary values, and grids that don't match. A bad structure is marked failed while the other structures still display.
- **Concurrency:** two jobs whose responses arrive in reverse order, with the same mask URL and different bytes, keep separate results.
- **Retries:** Retry-After honoured, 503 backoff, the attempt cap, non-retryable codes, `failed_auth_per_ip`, and network errors.
- **Token safety:** mask URLs on another host are refused, and a missing token sends nothing.
- **Unexpected exceptions:** a generic 500 with an error id; the traceback is logged server-side only.
- **Memory:** at most 3 series and 10 jobs are kept; the oldest are evicted.

## What I'd do next (Written by me not AI)

- Stream the volume, or decode it in a Web Worker, so larger series load progressively.
- Resample, with an explicit user opt-in, for masks on a different grid.
- Check headers against metadata using `GET /v1/cases/{id}`: shape, affine and the mask `sha256`.
- Features I think would add meaning to the user:
  - Add lesion-level metrics: per-component detection (TP/FP/FN lesions), Hausdorff 95 and NSD.
  - Persist jobs (SQLite or disk) so they survive a restart.
  - Add an end-to-end browser test (Playwright) for upload → overlay → Dice.
  - Add sagittal and coronal views, and a hover readout of HU and voxel position.

## AI tool use

1. As you know I am not a traditional developer so I relied heavily on Claude to construct this project
2. I prepared a list of requirements based on the task and saved it in the project directory as a markdown file to get direct instructions from.
3. A binary true/false environment file was also added that listed things that claude could and couldn't do. (E.g. git=FALSE)
4. I wrote a prompt to follow the instructions and guides.
5. ".env" -> saved token id, api path
