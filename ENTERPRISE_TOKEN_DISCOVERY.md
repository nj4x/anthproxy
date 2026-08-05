# Enterprise Token Discovery Guide

This guide helps you discover the format of your Anthropic enterprise OAuth token so we can implement fast prefix-based detection.

## Setup

1. **Start anthproxy with debug logging enabled:**

```bash
python -m anthproxy --backend anthropic --port 8082 --log-level DEBUG 2>&1 | tee anthproxy.log
```

This starts the server on port 8082 and logs all output (including our enterprise token discovery logs) to `anthproxy.log`.

2. **In another terminal, send a test request with your enterprise token:**

```bash
# Replace <your-enterprise-token> with an actual enterprise OAuth token from your Anthropic account
curl -X POST http://localhost:8082/v1/messages \
  -H "Authorization: Bearer <your-enterprise-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-haiku-20241022",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Say hello"}
    ]
  }'
```

3. **Inspect the logs:**

```bash
grep "Enterprise token discovery" anthproxy.log
```

Look for lines like:
```
Enterprise token discovery: Authorization=Bearer sk-ant-ent-..., x-api-key=(none)
```

## What to capture

From the logs, extract:
- **Full token prefix** — does it start with `sk-ant-ent-`, `sk-ant-`, something else?
- **Token length** — approximately how many characters?
- **Consistency** — do multiple enterprise tokens share the same prefix?

For comparison, test a **personal subscription token** (from your `~/.anthropic` file) to see if it has a different prefix:

```bash
# If you have personal anthropic creds, read the access token:
python -c "import json; print(json.load(open('~/.anthropic/access_tokens.json'))['access_token'][:40])"

# Then send a request with it:
curl -X POST http://localhost:8082/v1/messages \
  -H "Authorization: Bearer <your-personal-token>" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-haiku-20241022", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]}'

# Check logs again for the prefix
grep "Enterprise token discovery" anthproxy.log | tail -1
```

## Report findings

Once you have a test enterprise token and its prefix, update **#8** with:
1. Enterprise token prefix (e.g., `sk-ant-ent-...`)
2. Personal token prefix for comparison
3. Token length
4. Any other distinguishing characteristics

This information unblocks the implementation of fast detection logic in `prepare_routing()`.
