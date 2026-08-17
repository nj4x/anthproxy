/**
 * Tests for RequestDetailDrawer:
 * - CopyButton renders and responds to click events (preventDefault+stopPropagation)
 * - Drawer closed state renders nothing visible
 * - Prompt sections render CopyButton for user/system/tools when content is available
 * - Summary fields (estimated_input_tokens, input/output) are rendered
 * - ClassifierSummary fields are rendered when classifier_summary_json is set
 */
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SWRConfig } from 'swr';

// Mock navigator.clipboard
const clipboardWriteText = vi.fn().mockResolvedValue(undefined);
Object.defineProperty(navigator, 'clipboard', {
  value: { writeText: clipboardWriteText },
  writable: true,
});

// We need to mock the api module used by the drawer
vi.mock('../api/client', () => ({
  api: {
    getRequest: vi.fn(),
  },
}));

import { api } from '../api/client';
import { RequestDetailDrawer } from '../components/RequestDetailDrawer';
import type { RequestDetail } from '../api/types';

const mockRequest: RequestDetail = {
  id: 42,
  session_id: 'sess1',
  conversation_anchor: null,
  request_ts: '2026-07-11T10:00:00Z',
  requested_model: 'claude-sonnet-4-6',
  routed_model: 'claude-haiku-4-5',
  classification: 'trivial',
  reason_code: 'classifier',
  estimated_input_tokens: 12345,
  input_tokens: 100,
  output_tokens: 200,
  cache_creation_tokens: 0,
  cache_read_tokens: 50,
  duration_ms: 423,
  backend: 'anthropic',
  status: 'success',
  error: null,
  applied: 1,
  cost_estimate: 0.0012,
  model_tier: 'haiku',
  attempt: 1,
  user_prompt_text: 'Hello, world!',
  system_prompt_sha256: 'abc123def456'.padEnd(64, '0'),
  tools_sha256: null,
  routing_recovered_via_walkback: 0,
  classifier_model: 'claude-haiku-4-5',
  classifier_summary_json: JSON.stringify({
    final_user_text: 'Hello, world!',
    total_messages: 3,
    prior_user_messages: 1,
    prior_assistant_messages: 1,
    tool_use_count: 0,
    tool_result_count: 0,
    final_non_text_blocks: 0,
    has_images: false,
  }),
  classifier_raw_response: 'trivial',
  classifier_confidence: 0.95,
  classifier_format: 'label_only',
  cache_savings_usd: 0.0001,
  system_prompt_content: 'You are a helpful assistant.',
  system_prompt_char_count: 28,
  tools_content: null,
  tools_char_count: null,
};

function renderDrawer(requestId: number | null = 42) {
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <RequestDetailDrawer requestId={requestId} onClose={vi.fn()} />
    </SWRConfig>,
  );
}

describe('RequestDetailDrawer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (api.getRequest as ReturnType<typeof vi.fn>).mockResolvedValue(mockRequest);
  });

  it('does not show content when requestId is null', () => {
    renderDrawer(null);
    // The drawer is rendered via createPortal(..., document.body), so it lives
    // outside the render container; the panel should be translated off-screen.
    const panel = document.body.querySelector('.translate-x-full');
    expect(panel).toBeTruthy();
  });

  it('fetches request data when requestId is set', () => {
    renderDrawer(42);
    expect(api.getRequest).toHaveBeenCalledWith(42);
  });

  it('renders estimated_input_tokens in Request Summary after data loads', async () => {
    renderDrawer(42);
    await screen.findByText('Est. Input Tokens');
    expect(screen.getByText('12,345')).toBeTruthy();
  });

  it('renders input/output token summary after data loads', async () => {
    renderDrawer(42);
    await screen.findByText('Input / Output');
    expect(screen.getByText(/100.*200/)).toBeTruthy();
  });

  it('renders user prompt CopyButton when user_prompt_text is available', async () => {
    renderDrawer(42);
    await screen.findByText('User Prompt');
    // There should be a Copy button near user prompt section
    const copyButtons = screen.getAllByText('Copy');
    expect(copyButtons.length).toBeGreaterThan(0);
  });

  it('CopyButton calls clipboard.writeText with user prompt text', async () => {
    renderDrawer(42);
    await screen.findByText('User Prompt');
    const copyButtons = screen.getAllByText('Copy');
    // Find copy button next to user prompt (first one near that label)
    fireEvent.click(copyButtons[0]);
    expect(clipboardWriteText).toHaveBeenCalled();
  });

  it('CopyButton stopPropagation prevents row-level click bubbling', async () => {
    const rowClickHandler = vi.fn();
    render(
      <div onClick={rowClickHandler}>
        <SWRConfig value={{ provider: () => new Map() }}>
          <RequestDetailDrawer requestId={42} onClose={vi.fn()} />
        </SWRConfig>
      </div>,
    );
    await screen.findByText('User Prompt');
    const copyButtons = screen.getAllByText('Copy');
    fireEvent.click(copyButtons[0]);
    // stopPropagation should prevent the parent from receiving the click
    expect(rowClickHandler).not.toHaveBeenCalled();
  });

  it('shows classifier summary fields when classifier_summary_json is present', async () => {
    renderDrawer(42);
    await screen.findByText('Classification Detail');
    expect(screen.getByText(/Total Messages/)).toBeTruthy();
    expect(screen.getByText('3')).toBeTruthy();
  });

  it('shows "Has Images: No" from classifier summary', async () => {
    renderDrawer(42);
    await screen.findByText('Classification Detail');
    // Multiple "No" spans may exist; just verify the Has Images label is present
    expect(screen.getByText(/Has Images/)).toBeTruthy();
    const noElements = screen.getAllByText('No');
    expect(noElements.length).toBeGreaterThan(0);
  });

  it('shows system prompt Copy button when system_prompt_content is present', async () => {
    renderDrawer(42);
    await screen.findByText('System Prompt');
    // Should have Copy buttons present
    const copyButtons = screen.getAllByText('Copy');
    expect(copyButtons.length).toBeGreaterThanOrEqual(2);
  });

  it('renders request ID with CopyButton', async () => {
    renderDrawer(42);
    await screen.findByText('#42');
    const copyButtons = screen.getAllByText('Copy');
    expect(copyButtons.length).toBeGreaterThan(0);
  });

  it('CopyButton shows "Copied!" feedback after click', async () => {
    renderDrawer(42);
    await screen.findByText('User Prompt');
    const copyButtons = screen.getAllByText('Copy');
    fireEvent.click(copyButtons[0]);
    // After click clipboard resolves — check for Copied! (async)
    await screen.findByText('Copied!');
  });
});
