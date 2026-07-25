# Test images

A small set of real Iranian-plate photos for quickly trying Platrix — use them
in the **Image Detection** tab, the CLI, or the API.

```bash
# via the API (with an API token)
curl -H "Authorization: Bearer pltx_xxx" \
     -F "file=@img-test/sample-01.jpg" http://localhost:8080/api/recognize
```

The same images are mirrored on the model repo:
**https://huggingface.co/Dibachain/Platrix** (under `img-test/`).
