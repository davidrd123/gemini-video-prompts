# H3 Max video through fal

Use the generation MCP server's `start_fal_video_job`. This integration covers
`minimax/h3-max/image-to-video` and `minimax/h3-max/reference-to-video`.
The existing `generate_video` / `start_video_job` tools still use Seedance on
Replicate; the batch CLI's video mode still uses Gemini/Veo.

## Install and configure

From this checkout:

```bash
uv sync --locked --extra openai
uv run riff-mcp-doctor --provider fal
```

Keep `--extra openai` if you use the existing OpenAI integration. fal itself
requires no new SDK extra; it uses the already installed HTTP client. Configure
`FAL_KEY` in the untracked repository `.env` or process environment. Never print
or copy the key into tool calls, documentation, or shared client configuration.
The fal adapter reads `FAL_KEY`, not `FAL_API_KEY`. Existing environment values
win over `.env`. Do not add an empty client environment override.

The doctor checks local key presence and imports. It does not verify the key's
validity, model access, quota, or billing. `--network` reports that a fal auth
probe is not implemented; it does not generate a test video.

Restart the MCP client after updating. Use the same `gemini-prompts` entry in
[the example configuration](../.mcp.example.json). The refreshed server exposes
`start_fal_video_job` alongside the existing tools; verify it appears in the
client's tool list. See [the agent quickstart](AGENT_QUICKSTART.md) for setup.

## Choose an endpoint

| Input | Model | Fields |
|---|---|---|
| First frame, optionally a last frame | `minimax/h3-max/image-to-video` | `image`, optional `last_frame_image` |
| Ordered subject, style, motion, or sound references | `minimax/h3-max/reference-to-video` | `reference_images`, `reference_videos`, `reference_audios` |

Both accept integer durations from 5 to 15 seconds and `480p` or `768p`
(case insensitive; sent as `480P` / `768P`). Defaults are 5 seconds and 768p.
Each submission produces one video. These endpoints generate native audio;
they do not expose the Seedance `generate_audio` toggle.

Image-to-video follows the first frame's aspect ratio. Although fal's endpoint
can accept a missing image and route to text-to-video, riff requires a first
frame here to preserve the requested mode. Do not pass `aspect_ratio` or
`reference_*` inputs to this mode.

Reference-to-video supports at most 9 images, 3 videos, and 3 audio clips, with
12 files total. Each video/audio clip must be 2–15 seconds; the combined video
duration and combined audio duration must each be at most 15 seconds. Include
an image or video anchor; audio alone is insufficient. `ffprobe` must be on PATH
for video/audio validation. Riff rejects inputs it cannot validate.

Use `Image 1`, `Image 2`, `Video 1`, `Audio 1`, etc. in prompts. Numbering restarts
within each modality and follows list order. These are fal's tokens; they differ
from Seedance's bracket syntax. Reference-to-video `aspect_ratio` defaults to
`adaptive` and also accepts `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, and `9:16`.

Supply local file paths, preferably absolute. Riff sends the original bytes as
data URIs after approval and records their SHA-256 hashes. This first integration
accepts no remote reference URLs and limits inputs to 100 MiB total before base64
encoding. That is a local request-size limit, not a documented fal model limit.
Use smaller source clips if necessary; do not silently resize or trim references.

## Preview, approve, submit, and collect

Free MCP preview (the example path is a placeholder):

```json
{
  "prompt": "A blue ceramic cup stays still while a thin wisp of steam rises. Static camera, quiet room ambience.",
  "model": "minimax/h3-max/image-to-video",
  "image": "/ABSOLUTE/PATH/cup.png",
  "duration": 5,
  "resolution": "480p",
  "prompt_expansion_mode": "balanced",
  "title": "cup-steam",
  "dry_run": true
}
```

Call `start_fal_video_job` with this object. The preview validates parameter
shape and returns `billing_notice`, resolved parameters, and reference paths.
It does not read the key or media, write files, or contact fal. A valid preview
does not establish that files exist or that the key can generate videos.

Before execution, tell the user fal API billing applies. Obtain approval for
the endpoint, number of clips, duration, resolution, and references, including
possible reference-input charges. Honor an existing approval that covers that
scope. A saved key or auto-approved tool is not spending permission. Then repeat
the request with `dry_run=false` and `allow_api_billing=true`.

For reference-to-video, replace `model`, remove `image`, and supply an ordered
`reference_images` list (and optional video/audio lists). For example:

```json
{
  "prompt": "Image 1 is the protagonist; Image 2 is the setting. Preserve her appearance as she walks slowly through this space. Soft footsteps, no music.",
  "model": "minimax/h3-max/reference-to-video",
  "reference_images": ["/ABSOLUTE/PATH/character.png", "/ABSOLUTE/PATH/setting.png"],
  "duration": 5,
  "resolution": "768p",
  "prompt_expansion_mode": "balanced",
  "dry_run": true
}
```

Submission returns a local `job_id` and fal `request_id`. Call `get_video_job`
with that exact `job_id` and the same `out_root` if overridden. Each call checks
status once; when complete it retrieves the result and downloads the MP4.
Space polls several seconds apart. Inspect the returned `result.outputs` path
and view the video before judging creative success.

`get_video_job(..., poll=false)`, `get_generation`, and `list_generations` only
read local records. Repeated polling never submits another inference request.
`cancel_video_job` requests cancellation. Acceptance does not prove running
work stopped or that charges were avoided; continue checking the original job.

## Prompt expansion: balanced versus quality

fal documents `prompt_expansion_mode` as the effort spent rewriting the prompt
before generation:

| Mode | Documented rewrite latency | Use in riff |
|---|---|---|
| `balanced` | About one second | Default, matching fal |
| `quality` | Up to about 30 seconds for a richer prompt | Explicit per-call option |

The setting describes prompt preprocessing. The public schema does not say it
changes denoising steps or output resolution. fal's own prompting guide recommends
starting with balanced to retain the speed advantage. It supplies no controlled
balanced-versus-quality comparison. Our local same-seed pair is an exploratory
comparison, not a controlled benchmark establishing general superiority.

Our recommendation is to start with balanced and compare quality on a bounded
sample when added prompt detail might help. With carefully authored camera,
timing, and reference instructions, inspect the expanded text for unintended
changes. This is a workflow recommendation, not a measured superiority claim.

Riff preserves `prompt`, `resolved_params.prompt_expansion_mode`, and the returned
`expanded_prompt`. A null `expanded_prompt` does not prove no rewrite occurred:
fal says it can also mean the text was unchanged or expansion happened internally.
The current schemas describe balanced/quality but provide no explicit disabled
input option; riff does not advertise an unverified bypass. A repeated seed
alone is not a guarantee of identical output when rewritten text changes.

No separate expansion surcharge is listed on the endpoint pricing pages checked
2026-09-04. That is not a billing guarantee. `timings.inference` reports backend
denoising time when present, not total queue, rewrite, or download time.

## Billing, records, and recovery

Check live endpoint pricing before authorizing a batch. Resolution and output
seconds affect video charges. Reference-to-video also prices reference inputs
in a pooled token allowance; video references can materially change the total.
Do not estimate its bill from output duration alone. Launch discounts are
changing, so this guide deliberately links to live prices instead of making a
fixed rate part of the tool contract.

Local records use `out/jobs/<job_id>/request.json` and `status.json`, plus a
unique dated output folder and `job.json` after collection. They preserve original
request time, collection time, references/hashes, provider receipt/result,
expanded prompt, seed when returned, timings, output hash, and available ffprobe
media info. They are provenance, not a provider invoice or creative acceptance.
Large encoded input data and credentials are not written into request records.

A durable record exists before the submission POST. On an uncertain response,
the tool reports `FAL_SUBMISSION_UNCERTAIN` with its local `job_id`; inspect
`get_generation` and fal's dashboard. A crash can leave `status=submitting`.
Neither state proves the provider did or did not accept the request. Do not
resubmit without resolving that uncertainty and obtaining any needed approval.

There are no automatic HTTP retries or provider fallbacks in riff. fal's own
queue can retry work internally. HTTP 401/403 requires checking credentials or
access; 429 requires checking quota. A collection/download failure retains the
request ID; poll that existing job to retry collection without buying another
video. Riff accepts output downloads only from HTTPS fal.media hosts, limits a
file to 1 GiB locally, and does not send the API key to the media CDN.

Cancellation is best effort. A missing request after cancellation still needs
provider confirmation; riff does not interpret an arbitrary 404 as a refund or
proof of cancellation. `allow_api_billing` records caller acknowledgement and
is not independent proof of human consent or an enforced spending cap.

## Verification and sources

The integration has mocked HTTP coverage and two successful live 15-second,
768p reference-to-video runs with nine image references, using balanced and
quality expansion with the same returned seed. Both outputs measured 1344×768
at 24 fps with audio. Image-to-video, 480p, and video/audio reference inputs
remain offline-tested only. See [live verification](../LIVE_VERIFICATION.md)
for scope. Run the full offline suite and schema drift check:

```bash
uv run --extra openai pytest -q
uv run --extra openai ruff check src tests scripts
uv run python scripts/update_tool_reference.py --check
```

Primary sources checked 2026-09-04:

- [Image-to-video API and schema](https://fal.ai/models/minimax/h3-max/image-to-video/api)
- [Reference-to-video API and schema](https://fal.ai/models/minimax/h3-max/reference-to-video/api)
- [Image-to-video live pricing](https://fal.ai/models/minimax/h3-max/image-to-video)
- [Reference-to-video live pricing](https://fal.ai/models/minimax/h3-max/reference-to-video)
- [fal prompting and settings guide](https://fal.ai/learn/tools/how-to-use-minimax-h3-max)
- [fal queue lifecycle and cancellation](https://fal.ai/docs/documentation/model-apis/inference/queue)
