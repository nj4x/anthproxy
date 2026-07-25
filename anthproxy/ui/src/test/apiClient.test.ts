/**
 * Tests for api/client.ts additions:
 * - getSessionSummary exists and calls the correct endpoint
 * Tests for api/types.ts additions:
 * - ClassifierSummary, SessionSummary type shapes are used in tests
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { ClassifierSummary, SessionSummary } from '../api/types';

describe('ClassifierSummary type shape', () => {
  it('accepts all optional classifier summary fields', () => {
    const cs: ClassifierSummary = {
      final_user_text: 'Hello',
      total_messages: 5,
      prior_user_messages: 2,
      prior_assistant_messages: 2,
      tool_use_count: 1,
      tool_result_count: 1,
      final_non_text_blocks: 0,
      has_images: false,
      text_truncated: false,
      recovered_via_walkback: false,
    };
    expect(cs.final_user_text).toBe('Hello');
    expect(cs.has_images).toBe(false);
  });

  it('allows partial (all fields optional)', () => {
    const cs: ClassifierSummary = {};
    expect(cs.total_messages).toBeUndefined();
  });
});

describe('SessionSummary type shape', () => {
  it('has all required fields', () => {
    const ss: SessionSummary = {
      session_id: 'sess-abc',
      request_count: 10,
      total_input_tokens: 50000,
      total_output_tokens: 10000,
      total_cache_creation: 5000,
      total_cache_read: 45000,
      estimated_cost_usd: 1.23,
      pinned_backend: null,
      pinned_tier: 'sonnet',
    };
    expect(ss.session_id).toBe('sess-abc');
    expect(ss.pinned_tier).toBe('sonnet');
  });
});

describe('api.getSessionSummary', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    globalThis.fetch = fetchMock;
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  it('calls /admin/sessions/{id}/summary endpoint', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ session_id: 'sess1', request_count: 3 }),
    });

    const { api } = await import('../api/client');
    const result = await api.getSessionSummary('sess1');
    expect(fetchMock).toHaveBeenCalledWith(
      '/admin/sessions/sess1/summary',
      undefined,
    );
    expect(result.session_id).toBe('sess1');
  });

  it('URL-encodes special characters in session id', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ session_id: '{"session_id":"abc"}', request_count: 1 }),
    });

    const { api } = await import('../api/client');
    await api.getSessionSummary('{"session_id":"abc"}');
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('%7B%22session_id%22%3A%22abc%22%7D'),
      undefined,
    );
  });
});
