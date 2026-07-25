import { useEffect, useState } from 'react';
import useSWR from 'swr';
import { api } from '../api/client';
import type { RequestDetail, ClassifierSummary } from '../api/types';

function CopyButton({ text, title }: { text: string; title?: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };

  return (
    <button
      onClick={handleCopy}
      title={title ?? 'Copy to clipboard'}
      className="ml-2 px-1.5 py-0.5 rounded text-xs text-gray-400 hover:text-gray-700 hover:bg-gray-100 border border-transparent hover:border-gray-200 transition-colors"
    >
      {copied ? 'Copied!' : 'Copy'}
    </button>
  );
}

interface Props {
  requestId: number | null;
  onClose: () => void;
}

function reasonDescription(reasonCode: string | null): string {
  switch (reasonCode) {
    case 'size_forced_long_context':
      return 'No classifier call: long-context size floor applied';
    case 'affirmation_inherited':
      return 'No classifier call: short-affirmation (tier inherited)';
    case 'affirmation_floored_standard':
      return 'No classifier call: short-affirmation (floored to standard)';
    case 'session_cached_tier':
    case 'session_cached_walkback':
      return 'No classifier call: cached tier replayed';
    case 'missing_final_user_text':
      return 'No classifier call: no user text in final message';
    case 'override_no_classifier':
      return 'No classifier call: X-Anthproxy-Override: no-classifier';
    default:
      return 'No classifier call';
  }
}

function ClassificationChip({ cls }: { cls: string | null }) {
  if (!cls) return <span className="text-gray-400">—</span>;
  const colorCls =
    cls === 'trivial' ? 'bg-green-100 text-green-800' :
    cls === 'standard' ? 'bg-amber-100 text-amber-800' :
    cls === 'deep' ? 'bg-red-100 text-red-800' :
    'bg-gray-100 text-gray-800';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${colorCls}`}>
      {cls}
    </span>
  );
}

function StatusChip({ status }: { status: string }) {
  const cls =
    status === 'success' ? 'bg-green-100 text-green-800' :
    status === 'rate_limited' ? 'bg-amber-100 text-amber-800' :
    'bg-red-100 text-red-800';
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {status}
    </span>
  );
}

function cacheHitRatio(d: RequestDetail): string {
  const cr = d.cache_read_tokens ?? 0;
  const inp = d.input_tokens ?? 0;
  const denom = inp + cr;
  if (denom === 0) return '—';
  return (cr / denom * 100).toFixed(1) + '%';
}

function prettyJson(raw: string | null | undefined): string {
  if (!raw) return '';
  try {
    return JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    return raw;
  }
}

export function RequestDetailDrawer({ requestId, onClose }: Props) {
  const { data, error, isLoading, mutate } = useSWR<RequestDetail>(
    requestId != null ? ['request-detail', requestId] : null,
    () => api.getRequest(requestId!),
  );

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [onClose]);

  const isOpen = requestId != null;

  const parsedSummary = (() => {
    if (!data?.classifier_summary_json) return null;
    try {
      return JSON.parse(data.classifier_summary_json) as ClassifierSummary;
    } catch {
      return null;
    }
  })();

  const modelsDiffer =
    data != null &&
    data.routed_model != null &&
    data.routed_model !== data.requested_model;

  return (
    <>
      <div
        className={`fixed inset-0 bg-black z-40 transition-opacity duration-200 ${isOpen ? 'opacity-25' : 'opacity-0 pointer-events-none'}`}
        onClick={onClose}
      />

      <div
        className={`fixed top-0 right-0 h-full w-[600px] max-w-full bg-white shadow-2xl z-50 overflow-y-auto transform transition-transform duration-200 ${isOpen ? 'translate-x-0' : 'translate-x-full'}`}
      >
        <div className="sticky top-0 bg-white border-b border-gray-200 px-5 py-3 flex items-center justify-between z-10">
          <span className="text-sm font-semibold text-gray-800">
            {data ? `Request #${data.id}` : 'Request Detail'}
          </span>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-xl leading-none w-8 h-8 flex items-center justify-center"
          >
            ×
          </button>
        </div>

        <div className="px-5 py-4 space-y-6 text-sm">
          {isLoading && (
            <div className="text-gray-500">Loading...</div>
          )}

          {error && !isLoading && (
            <div className="bg-red-50 text-red-700 p-4 rounded">
              {error.message}
              <button
                onClick={() => mutate()}
                className="ml-3 text-red-600 underline text-xs"
              >
                Retry
              </button>
            </div>
          )}

          {data && (
            <>
              {/* 1. Request Summary */}
              <section>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Request Summary</h3>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Time</dt>
                    <dd className="text-gray-800 text-sm">{new Date(data.request_ts).toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Backend</dt>
                    <dd className="text-gray-800 text-sm">{data.backend}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Routed Model</dt>
                    <dd className="text-gray-800 font-mono text-xs">{data.routed_model ?? '—'}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Status</dt>
                    <dd><StatusChip status={data.status} /></dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Duration</dt>
                    <dd className="text-gray-800 text-sm">
                      {data.duration_ms != null ? `${data.duration_ms.toLocaleString()}ms` : '—'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Request ID</dt>
                    <dd className="flex items-center">
                      <span className="text-gray-800 font-mono text-xs">#{data.id}</span>
                      <CopyButton text={String(data.id)} title="Copy request ID" />
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Est. Input Tokens</dt>
                    <dd className="text-gray-800 text-sm">
                      {data.estimated_input_tokens != null ? data.estimated_input_tokens.toLocaleString() : '—'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Input / Output</dt>
                    <dd className="text-gray-800 text-sm">
                      {data.input_tokens != null ? data.input_tokens.toLocaleString() : '—'}
                      {' / '}
                      {data.output_tokens != null ? data.output_tokens.toLocaleString() : '—'}
                    </dd>
                  </div>
                </dl>
              </section>

              {/* 2. Routing Decision */}
              <section>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Routing Decision</h3>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Requested Model</dt>
                    <dd className="text-gray-800 font-mono text-xs">{data.requested_model}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Routed Model</dt>
                    <dd className={`font-mono text-xs ${modelsDiffer ? 'bg-yellow-50 text-yellow-800 px-1 rounded' : 'text-gray-800'}`}>
                      {data.routed_model
                        ? (modelsDiffer ? `→ ${data.routed_model}` : data.routed_model)
                        : '—'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Classification</dt>
                    <dd><ClassificationChip cls={data.classification} /></dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Reason Code</dt>
                    <dd className="font-mono text-xs text-gray-700">{data.reason_code ?? '—'}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Applied</dt>
                    <dd className="text-gray-800 text-sm">{data.applied === 1 ? 'Yes' : 'No'}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Model Tier</dt>
                    <dd className="text-gray-800 text-sm">{data.model_tier ?? '—'}</dd>
                  </div>
                </dl>
              </section>

              {/* 3. Prompt Caching */}
              <section>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Prompt Caching</h3>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-3">
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Cache Read Tokens</dt>
                    <dd className="text-gray-800 text-sm">{data.cache_read_tokens?.toLocaleString() ?? '—'}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Cache Creation Tokens</dt>
                    <dd className="text-gray-800 text-sm">{data.cache_creation_tokens?.toLocaleString() ?? '—'}</dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Cache Savings</dt>
                    <dd className="text-gray-800 text-sm">
                      {data.cache_savings_usd != null ? `$${data.cache_savings_usd.toFixed(4)}` : '—'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-xs text-gray-400 mb-0.5">Cache Hit Ratio</dt>
                    <dd className="text-gray-800 text-sm">{cacheHitRatio(data)}</dd>
                  </div>
                </dl>
              </section>

              {/* 4. Prompt Content */}
              {(data.user_prompt_text !== null || data.system_prompt_sha256 !== null || data.tools_sha256 !== null) && (
                <section>
                  <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Prompt Content</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="text-xs text-gray-500 font-medium mb-1 flex items-center">
                        User Prompt
                        {data.user_prompt_text != null && (
                          <CopyButton text={prettyJson(data.user_prompt_text)} title="Copy user prompt" />
                        )}
                      </div>
                      {data.user_prompt_text != null ? (
                        <textarea
                          readOnly
                          value={prettyJson(data.user_prompt_text)}
                          className="w-full font-mono text-xs border border-gray-200 rounded p-2 resize-none bg-gray-50 overflow-y-auto"
                          style={{ maxHeight: '400px', minHeight: '80px' }}
                          rows={6}
                        />
                      ) : (
                        <span className="text-xs text-gray-400 italic">— (no text content in final message)</span>
                      )}
                    </div>

                    <div>
                      <div className="text-xs text-gray-500 font-medium mb-1 flex items-center">
                        System Prompt
                        {data.system_prompt_char_count != null && (
                          <span className="text-gray-400 font-normal ml-1">({data.system_prompt_char_count.toLocaleString()} chars)</span>
                        )}
                        {data.system_prompt_content != null && (
                          <CopyButton text={prettyJson(data.system_prompt_content)} title="Copy system prompt" />
                        )}
                      </div>
                      {data.system_prompt_sha256 != null ? (
                        <details>
                          <summary className="text-xs text-blue-600 cursor-pointer select-none">Show content</summary>
                          <pre className="mt-2 font-mono text-xs bg-gray-50 border border-gray-200 rounded p-2 overflow-auto max-h-64 whitespace-pre-wrap break-words">
                            {data.system_prompt_content ? prettyJson(data.system_prompt_content) : '(content not stored)'}
                          </pre>
                        </details>
                      ) : (
                        <span className="text-xs text-gray-400">None</span>
                      )}
                    </div>

                    <div>
                      <div className="text-xs text-gray-500 font-medium mb-1 flex items-center">
                        Tools
                        {data.tools_char_count != null && (
                          <span className="text-gray-400 font-normal ml-1">({data.tools_char_count.toLocaleString()} chars)</span>
                        )}
                        {data.tools_content != null && (
                          <CopyButton text={prettyJson(data.tools_content)} title="Copy tools JSON" />
                        )}
                      </div>
                      {data.tools_sha256 != null ? (
                        <details>
                          <summary className="text-xs text-blue-600 cursor-pointer select-none">Show content</summary>
                          <pre className="mt-2 font-mono text-xs bg-gray-50 border border-gray-200 rounded p-2 overflow-auto max-h-64 whitespace-pre-wrap break-words">
                            {data.tools_content ? prettyJson(data.tools_content) : '(content not stored)'}
                          </pre>
                        </details>
                      ) : (
                        <span className="text-xs text-gray-400">None (0 tools)</span>
                      )}
                    </div>
                  </div>
                </section>
              )}

              {/* 5. LLM Response */}
              {data.response_text != null && (
                <section>
                  <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">
                    LLM Response
                  </h3>
                  <div>
                    <div className="text-xs text-gray-500 font-medium mb-1 flex items-center">
                      Response Text
                      <CopyButton text={data.response_text} title="Copy response text" />
                    </div>
                    <textarea
                      readOnly
                      value={data.response_text}
                      className="w-full font-mono text-xs border border-gray-200 rounded p-2 resize-none bg-gray-50 overflow-y-auto"
                      style={{ maxHeight: '400px', minHeight: '80px' }}
                      rows={6}
                    />
                  </div>
                </section>
              )}

              {/* 6. Classification Detail */}
              <section>
                <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-3">Classification Detail</h3>

                {data.classifier_model != null ? (
                  <div className="space-y-3">
                    <div className="text-xs text-gray-600 font-medium">
                      Classifier: <span className="font-mono">{data.classifier_model}</span>
                      {data.classifier_format && (
                        <span className="text-gray-400 font-normal"> ({data.classifier_format})</span>
                      )}
                    </div>

                    <div className="text-xs">
                      <span className="text-gray-400">Walk-back: </span>
                      <span className="text-gray-700">
                        {data.routing_recovered_via_walkback === 1
                          ? 'Yes (walk-back text source)'
                          : 'No'}
                      </span>
                    </div>

                    {parsedSummary && (
                      <div className="bg-gray-50 rounded p-3 space-y-3 text-xs">
                        {parsedSummary.final_user_text && (
                          <div>
                            <div className="text-gray-400 mb-1 flex items-center">
                              User text (classifier input)
                              <CopyButton text={parsedSummary.final_user_text} title="Copy classifier input text" />
                            </div>
                            <blockquote className="border-l-2 border-gray-300 pl-3 text-gray-700 font-mono whitespace-pre-wrap break-words">
                              {parsedSummary.final_user_text.slice(0, 500)}
                              {parsedSummary.text_truncated === true && (
                                <span className="text-gray-400"> … (truncated)</span>
                              )}
                            </blockquote>
                          </div>
                        )}
                        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                          {([
                            ['Total Messages', parsedSummary.total_messages],
                            ['Prior User', parsedSummary.prior_user_messages],
                            ['Prior Assistant', parsedSummary.prior_assistant_messages],
                            ['Tool Use', parsedSummary.tool_use_count],
                            ['Tool Results', parsedSummary.tool_result_count],
                            ['Non-text Blocks', parsedSummary.final_non_text_blocks],
                            ['Has Images', parsedSummary.has_images === true ? 'Yes' : parsedSummary.has_images === false ? 'No' : undefined],
                          ] as [string, unknown][])
                            .filter(([, v]) => v !== undefined && v !== null)
                            .map(([label, val]) => (
                              <div key={label as string}>
                                <span className="text-gray-400">{label as string}: </span>
                                <span className="text-gray-700">{String(val)}</span>
                              </div>
                            ))}
                        </div>
                      </div>
                    )}

                    {data.classifier_raw_response && (
                      <div>
                        <div className="text-xs text-gray-400 mb-1">Classifier Response</div>
                        <pre
                          className="font-mono text-xs bg-gray-50 border border-gray-200 rounded p-2 overflow-y-auto whitespace-pre-wrap break-words"
                          style={{ maxHeight: '200px' }}
                        >
                          {data.classifier_raw_response}
                        </pre>
                      </div>
                    )}

                    <div className="text-xs">
                      <span className="text-gray-400">Confidence: </span>
                      <span className="text-gray-700">{data.classifier_confidence?.toFixed(2) ?? '—'}</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-gray-600">{reasonDescription(data.reason_code)}</p>
                )}
              </section>
            </>
          )}
        </div>
      </div>
    </>
  );
}
