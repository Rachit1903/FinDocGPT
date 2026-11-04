import { useMemo, useState } from 'react';

const SAMPLE_QUERIES = [
  'What did Apple identify as the main drivers of Services net sales in fiscal 2025?',
  "Compare Google Services and Google Cloud operating income trends.",
  "Calculate Google Cloud's operating-margin change between the two latest annual periods.",
];

function Citations({ text }) {
  // Render "[S3]" markers as small badges.
  const parts = text.split(/(\[S\d+\])/g);
  return parts.map((part, i) =>
    /^\[S\d+\]$/.test(part) ? <sup key={i} className="cite">{part.slice(1, -1)}</sup> : <span key={i}>{part}</span>,
  );
}

function SourceCard({ source, open }) {
  const meta = [source.ticker, source.fiscal_year && `FY${source.fiscal_year}`, source.item, source.subsection]
    .filter(Boolean)
    .join(' · ');
  return (
    <details className="source-card" open={open}>
      <summary>
        <div>
          <span className="source-pill">{source.label}</span>
          <strong>{source.chunk_id}</strong>
          {meta ? <p>{meta}</p> : null}
        </div>
        <div className="source-stats">
          <span>{source.kind}</span>
          {typeof source.rerank_score === 'number' ? <span>rerank {source.rerank_score.toFixed(2)}</span> : null}
        </div>
      </summary>
      <pre className="source-content">{source.text}</pre>
    </details>
  );
}

function Trace({ trace }) {
  return (
    <ol className="trace">
      {trace.map((step, i) => (
        <li key={i}>
          <strong>{step.step}</strong> <span className="muted">{step.ms} ms</span>
          {step.step === 'plan' ? <p>route: {step.route} · companies: {step.tickers.join(', ') || '—'} · {step.sub_queries.length} sub-queries</p> : null}
          {step.step === 'rewrite' ? <p>rewrote {Object.keys(step.rewrites).length} weak sub-quer{Object.keys(step.rewrites).length === 1 ? 'y' : 'ies'}</p> : null}
          {step.step === 'calculate' ? <p>{step.result || 'no grounded calculation'} ({step.attempts.length} attempt{step.attempts.length === 1 ? '' : 's'})</p> : null}
          {step.step === 'verify' ? <p>{step.kept} supported, {step.removed} rejected{step.revising ? ' → sent back for revision' : ''}</p> : null}
          {step.step === 'revise' ? <p>rewrote the answer to address {step.objections} objection(s)</p> : null}
        </li>
      ))}
    </ol>
  );
}

export default function App() {
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [response, setResponse] = useState(null);
  const apiBase = useMemo(() => (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, ''), []);

  async function runQuery() {
    const trimmed = query.trim();
    if (!trimmed || loading) return;
    setLoading(true);
    setError('');
    try {
      const res = await fetch(`${apiBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed }),
      });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload?.detail || 'Request failed.');
      setResponse(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <main className="layout">
        <section className="hero panel">
          <h1>FinDocGPT</h1>
          <p className="lede">
            Ask analyst questions about the latest 10-K filings of ten large-cap companies. An agent plans the research,
            retrieves and reranks filing excerpts, runs a grounded calculator when numbers are needed, and removes any
            sentence it cannot verify against the sources.
          </p>
          <form
            className="query-card"
            onSubmit={(event) => {
              event.preventDefault();
              runQuery();
            }}
          >
            <label htmlFor="query">Ask a filing question</label>
            <div className="input-wrapper">
              <textarea
                id="query"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    runQuery();
                  }
                }}
                rows={3}
              />
              <button className="icon-button" type="submit" disabled={loading} title="Ask">
                {loading ? '…' : '↑'}
              </button>
            </div>
            <div className="sample-row">
              {SAMPLE_QUERIES.map((sample) => (
                <button key={sample} type="button" className="sample-chip" onClick={() => setQuery(sample)}>
                  {sample}
                </button>
              ))}
            </div>
          </form>
        </section>

        <section className="answer-column">
          <article className="panel answer-panel">
            <div className="panel-header">
              <div>
                <div className="eyebrow">Answer</div>
                <h2>{response ? `Route: ${response.route}` : loading ? 'Researching…' : 'Waiting for a question'}</h2>
              </div>
            </div>
            {error ? <div className="error-box">{error}</div> : null}
            {response ? (
              <>
                <div className="answer-block">
                  <p><Citations text={response.answer} /></p>
                </div>
                {response.calculation ? (
                  <div className="computation-card">
                    <span className="label">Calculator</span>
                    <p><strong>{response.calculation.label}: {response.calculation.formatted}</strong></p>
                    <p className="muted">{response.calculation.expression} where{' '}
                      {response.calculation.operands.map((o) => `${o.name} = ${o.value_text} (${o.label})`).join('; ')}</p>
                  </div>
                ) : null}
                {response.removed_claims.length ? (
                  <div className="answer-block subdued">
                    <span className="label">Removed by verification</span>
                    <ul>{response.removed_claims.map((c) => <li key={c}>{c}</li>)}</ul>
                  </div>
                ) : null}
                <details className="answer-block subdued">
                  <summary className="label">Agent trace</summary>
                  <Trace trace={response.trace} />
                </details>
              </>
            ) : (
              <div className="empty-state"><p>The verified answer, calculator output and agent trace appear here.</p></div>
            )}
          </article>

          <article className="panel sources-panel">
            <div className="panel-header">
              <div>
                <div className="eyebrow">Citations</div>
                <h2>Source excerpts</h2>
              </div>
              <span className="counter">{response?.sources?.length || 0} sources</span>
            </div>
            {response?.sources?.length ? (
              <div className="source-list">
                {response.sources.map((source, index) => (
                  <SourceCard key={source.chunk_id} source={source} open={index === 0} />
                ))}
              </div>
            ) : (
              <div className="empty-state compact">Sources appear after a question runs.</div>
            )}
          </article>
        </section>
      </main>
    </div>
  );
}
