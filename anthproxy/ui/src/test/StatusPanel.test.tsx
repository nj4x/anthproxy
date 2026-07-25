import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SWRConfig } from 'swr';

vi.mock('../api/client', () => ({
  api: { getStatus: vi.fn() },
}));

import { api } from '../api/client';
import { StatusPanel } from '../components/StatusPanel';
import type { StatusResponse } from '../api/types';

function statusWithUsage(subscription_usage: StatusResponse['subscription_usage']): StatusResponse {
  return {
    active_backend: 'anthropic',
    routing_enabled: false,
    routing_mode: 'off',
    classifier_model: '',
    long_context_threshold: 0,
    affirmation_inherit: false,
    backends: [],
    session_overrides: [],
    subscription_usage,
  };
}

function renderStatus(data: StatusResponse) {
  (api.getStatus as ReturnType<typeof vi.fn>).mockResolvedValue(data);
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <StatusPanel />
    </SWRConfig>,
  );
}

describe('StatusPanel usage meters', () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('uses the existing red-head calculation for Anthropic five-hour usage', async () => {
    const { container } = renderStatus(statusWithUsage({
      anthropic: {
        five_hour: { used_tokens: null, limit_tokens: null, pct: 95, reset_at: null, reset_in_secs: 3600, window_hours: 5 },
      },
    }));

    await screen.findByText('5-hour window');
    const redHead = Array.from(container.querySelectorAll('.bg-red-500')).find(
      (element) => element.getAttribute('style')?.includes('left: 80%'),
    );
    expect(redHead).toHaveStyle({ left: '80%', width: '15%' });
  });

  it('labels Codex rolling usage by its true duration and renders a red head', async () => {
    const { container } = renderStatus(statusWithUsage({
      codex: {
        weekly: { used_tokens: null, limit_tokens: null, pct: 95, reset_at: null, reset_in_secs: 24 * 3600, window_hours: 168 },
      },
    }));

    await screen.findByText('168-hour window');
    expect(screen.queryByText('Weekly')).not.toBeInTheDocument();
    const redHead = Array.from(container.querySelectorAll('.bg-red-500')).find(
      (element) => element.getAttribute('style')?.includes('left:'),
    );
    expect(redHead).toBeDefined();
    expect(redHead?.getAttribute('style')).toContain('width:');
  });

  it('does not render a red head without a reset countdown', async () => {
    const { container } = renderStatus(statusWithUsage({
      anthropic: {
        five_hour: { used_tokens: null, limit_tokens: null, pct: 95, reset_at: null, reset_in_secs: null, window_hours: 5 },
      },
    }));

    await screen.findByText('5-hour window');
    const redHead = Array.from(container.querySelectorAll('.bg-red-500')).find(
      (element) => element.getAttribute('style')?.includes('left:'),
    );
    expect(redHead).toBeUndefined();
  });

  it('renders a green head when usage is below linear pace', async () => {
    // 1 hour left of a 5-hour window → redStart = 100 - (3600/3600)*(100/5) = 80
    // pct=50 < redStart=80 → green segment [50%, 80%], width=30%
    const { container } = renderStatus(statusWithUsage({
      anthropic: {
        five_hour: { used_tokens: null, limit_tokens: null, pct: 50, reset_at: null, reset_in_secs: 3600, window_hours: 5 },
      },
    }));

    await screen.findByText('5-hour window');
    const greenHead = Array.from(container.querySelectorAll('.bg-green-500')).find(
      (element) => element.getAttribute('style')?.includes('left: 50%'),
    );
    expect(greenHead).toHaveStyle({ left: '50%', width: '30%' });
  });

  it('does not render a green head when usage is above linear pace (red head territory)', async () => {
    // pct=95 > redStart=80 → red head; no green head
    const { container } = renderStatus(statusWithUsage({
      anthropic: {
        five_hour: { used_tokens: null, limit_tokens: null, pct: 95, reset_at: null, reset_in_secs: 3600, window_hours: 5 },
      },
    }));

    await screen.findByText('5-hour window');
    const greenHead = Array.from(container.querySelectorAll('.bg-green-500')).find(
      (element) => element.getAttribute('style')?.includes('left:'),
    );
    expect(greenHead).toBeUndefined();
  });

  it('does not render a green head at the exact pace boundary', async () => {
    // pct=80 === redStart=80 → neither head (redStart - pct = 0, below 0.5 threshold)
    const { container } = renderStatus(statusWithUsage({
      anthropic: {
        five_hour: { used_tokens: null, limit_tokens: null, pct: 80, reset_at: null, reset_in_secs: 3600, window_hours: 5 },
      },
    }));

    await screen.findByText('5-hour window');
    const greenHead = Array.from(container.querySelectorAll('.bg-green-500')).find(
      (element) => element.getAttribute('style')?.includes('left:'),
    );
    expect(greenHead).toBeUndefined();
  });

  it('does not render a green head when there is no reset countdown', async () => {
    // reset_in_secs null → redStart cannot be computed → no green head
    const { container } = renderStatus(statusWithUsage({
      anthropic: {
        five_hour: { used_tokens: null, limit_tokens: null, pct: 50, reset_at: null, reset_in_secs: null, window_hours: 5 },
      },
    }));

    await screen.findByText('5-hour window');
    const greenHead = Array.from(container.querySelectorAll('.bg-green-500')).find(
      (element) => element.getAttribute('style')?.includes('left:'),
    );
    expect(greenHead).toBeUndefined();
  });
});
