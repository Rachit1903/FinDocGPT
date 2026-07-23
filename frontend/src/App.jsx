import { useMemo, useState } from 'react';

const SAMPLE_QUERIES = [
  "What did Apple say about its services business?",
  'Compare Apple and Microsoft gross-margin discussion.',
  "Calculate how Apple's total gross margin percentage changed from 2024 to 2025.",
];

function formatMeta(metadata) {
  const fields = ['ticker', 'filing_type', 'filing_date', 'section'];
  return fields
    .map((field) => metadata?.[field])
    .filter(Boolean)
    .join(' · ');
}

function SourceCard({ source, index }) {
  const meta = formatMeta(source.metadata);

  return (
    <details className="source-card" open={index === 0}>
      <summary>
        <div>
          <span className="source-pill">Source {index + 1}</span>
          <strong>{source.chunk_id}</strong>
          {meta ? <p>{meta}</p> : null}
        </div>
        <div className="source-stats">
          {source.reranked_rank ? <span>Rank #{source.reranked_rank}</span> : null}
          {typeof source.rrf_score === 'number' ? <span>RRF {source.rrf_score.toFixed(3)}</span> : null}
        </div>
      </summary>
      <pre>{source.text}</pre>
    </details>
  );
}

function VerdictSummary({ verdicts }) {
  const supported = verdicts.filter((verdict) => verdict.supported);
  const unsupported = verdicts.filter((verdict) => !verdict.supported);

  return (
    <div className="verdict-grid">
      <div className="verdict-box success">
        <span>Supported claims</span>
        <strong>{supported.length}</strong>
      </div>
      <div className="verdict-box warning">
        <span>Removed claims</span>
        <strong>{unsupported.length}</strong>
      </div>
      {unsupported.length ? <p className="verdict-note">Unsupported claims were stripped before the answer was returned.</p> : null}
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [response, setResponse] = useState(null);

  const apiBase = useMemo(() => (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, ''), []);

  async function runQuery(nextQuery = query) {
    const trimmed = nextQuery.trim();
    if (!trimmed || loading) {
      return;
    }

    setLoading(true);
    setError('');

    try {
      const res = await fetch(`${apiBase}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: trimmed, k: 5 }),
      });

      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload?.detail || 'Request failed.');
      }

      setResponse(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected error.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <main className="layout">
        <section className="hero panel">
          <div className="eyebrow">FinDocGPT · Phase 7</div>
          <h1>Agentic SEC research with citations you can inspect.</h1>
          <p className="lede">
            Query the hybrid retriever, reranker, and numeric reasoning path from a single chat UI, then expand the
            supporting filing excerpts below each answer.
          </p>

          <form
            className="query-card"
            onSubmit={(event) => {
              event.preventDefault();
              runQuery();
            }}
          >
            <label htmlFor="query">Ask a filing question</label>
            <textarea
              id="query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              rows={4}
              placeholder="Ask about margin, revenue drivers, comparisons, or a computed metric."
            />

            <div className="query-actions">
              <div className="sample-row">
                {SAMPLE_QUERIES.map((sample) => (
                  <button key={sample} type="button" className="sample-chip" onClick={() => setQuery(sample)}>
                    {sample}
                  </button>
                ))}
              </div>
              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? 'Answering...' : 'Run FinDocGPT'}
              </button>
            </div>
          </form>

          <div className="quick-facts">
            <div>
              <span>Routing</span>
              <strong>Factual, comparison, or computation</strong>
            </div>
            <div>
              <span>Retrieval</span>
              <strong>BM25 + dense RRF + reranker</strong>
            </div>
            <div>
              <span>Verification</span>
              <strong>Claims stripped unless supported</strong>
            </div>
          </div>
        </section>

        <section className="answer-column">
          <article className="panel answer-panel">
            <div className="panel-header">
              <div>
                <div className="eyebrow">Answer</div>
                <h2>{response ? 'Verified result' : 'Waiting for a query'}</h2>
              </div>
              {response ? <span className={`route-badge route-${response.route}`}>{response.route}</span> : null}
            </div>

            {error ? <div className="error-box">{error}</div> : null}

            {response ? (
              <>
                <p className="route-reason">{response.route_reason}</p>
                <div className="answer-block">
                  <span className="label">Returned answer</span>
                  <p>{response.answer}</p>
                </div>

                {response.computation ? (
                  <div className="computation-card">
                    <span className="label">Numeric computation</span>
                    <p>{response.computation.calculation}</p>
                    <div className="metric-grid">
                      <div>
                        <span>Table</span>
                        <strong>{response.computation.table_id}</strong>
                      </div>
                      <div>
                        <span>Ticker</span>
                        <strong>{response.computation.ticker}</strong>
                      </div>
                      <div>
                        <span>Metric</span>
                        <strong>{response.computation.metric}</strong>
                      </div>
                    </div>
                  </div>
                ) : null}

                <div className="answer-block subdued">
                  <span className="label">Draft answer before verification</span>
                  <p>{response.draft_answer}</p>
                </div>

                {response.verdicts?.length ? <VerdictSummary verdicts={response.verdicts} /> : null}
              </>
            ) : (
              <div className="empty-state">
                <p>The backend will return the verified answer, the reasoning path, and the supporting source chunks.</p>
              </div>
            )}
          </article>

          <article className="panel sources-panel">
            <div className="panel-header">
              <div>
                <div className="eyebrow">Citations</div>
                <h2>Expandable source excerpts</h2>
              </div>
              <span className="counter">{response?.sources?.length || 0} sources</span>
            </div>

            {response?.sources?.length ? (
              <div className="source-list">
                {response.sources.map((source, index) => (
                  <SourceCard key={`${source.chunk_id}-${index}`} source={source} index={index} />
                ))}
              </div>
            ) : (
              <div className="empty-state compact">
                Source chunks will appear here after a query runs.
              </div>
            )}
          </article>
        </section>
      </main>
    </div>
  );
}
