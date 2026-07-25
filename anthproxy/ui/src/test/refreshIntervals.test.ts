/**
 * Verify that every SWR hook in the permitted files uses refreshInterval: 60000.
 * This test reads the source files directly to assert the configuration.
 */
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { describe, it, expect } from 'vitest';

const root = resolve(__dirname, '../..');

function readSrc(relPath: string): string {
  return readFileSync(resolve(root, 'src', relPath), 'utf-8');
}

describe('SWR refreshInterval — all should be 60000', () => {
  it('App.tsx layout-config uses 60000', () => {
    const src = readSrc('App.tsx');
    expect(src).not.toContain('refreshInterval: 30000');
    expect(src).toMatch(/refreshInterval:\s*60000/);
  });

  it('Dashboard.tsx all three hooks use 60000', () => {
    const src = readSrc('pages/Dashboard.tsx');
    const matches = src.match(/refreshInterval:\s*60000/g) ?? [];
    expect(matches.length).toBeGreaterThanOrEqual(3);
    expect(src).not.toContain('refreshInterval: 30000');
  });

  it('Sessions.tsx uses 60000', () => {
    const src = readSrc('pages/Sessions.tsx');
    expect(src).not.toContain('refreshInterval: 30000');
    expect(src).toMatch(/refreshInterval:\s*60000/);
  });

  it('SessionDetail.tsx session hook uses 60000', () => {
    const src = readSrc('pages/SessionDetail.tsx');
    expect(src).not.toContain('refreshInterval: 30000');
    const matches = src.match(/refreshInterval:\s*60000/g) ?? [];
    // Expect at least two: session + trace
    expect(matches.length).toBeGreaterThanOrEqual(2);
  });

  it('StatusPanel.tsx uses 60000', () => {
    const src = readSrc('components/StatusPanel.tsx');
    expect(src).not.toContain('refreshInterval: 30000');
    expect(src).toMatch(/refreshInterval:\s*60000/);
  });
});
