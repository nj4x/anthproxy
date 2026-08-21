/**
 * Cost totals are scoped to the recording instance (ADR-0025).
 *
 * A chained request is recorded by every hop that handled it, under the same
 * session key, so adding one instance's total to another's double-counts it.
 */
export function CostScopeNote({ className = '' }: { className?: string }) {
  return (
    <p className={`text-xs text-gray-400 ${className}`}>
      Costs cover traffic this instance handled. Requests dispatched to a{' '}
      <code className="font-mono">peer</code> are recorded here <em>and</em> at the peer, so
      totals from chained instances are not additive.
    </p>
  );
}
