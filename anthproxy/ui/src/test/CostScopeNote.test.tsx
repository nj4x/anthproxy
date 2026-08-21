import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { CostScopeNote } from '../components/CostScopeNote';

afterEach(cleanup);

describe('CostScopeNote', () => {
  it('states that chained instances are not additive', () => {
    render(<CostScopeNote />);
    expect(screen.getByText(/not additive/i)).toBeInTheDocument();
    expect(screen.getByText('peer')).toBeInTheDocument();
  });

  it('keeps caller-supplied layout classes', () => {
    const { container } = render(<CostScopeNote className="mb-6" />);
    expect(container.querySelector('p')).toHaveClass('mb-6');
  });
});
