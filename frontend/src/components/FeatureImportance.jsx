import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

export default function FeatureImportance({ result }) {
  if (!result) return null
  const { local_contributions, global_importance, natural_language_explanation, recommendations } = result

  // Local SHAP chart data (top 10-12)
  const localData = (local_contributions || [])
    .filter(c => typeof c.shap_value === 'number')
    .slice(0, 10)
    .map(c => ({
      name: c.display_name || c.feature,
      value: c.shap_value,
      significance: c.significance,
    }))

  // Global SHAP data
  const globalData = (global_importance?.shap_global || [])
    .slice(0, 8)
    .map(c => ({
      name: c.display_name || c.feature,
      value: c.mean_abs_shap,
    }))

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null
    const d = payload[0]
    const isLocal = typeof d.payload.value === 'number' && d.payload.significance !== undefined

    return (
      <div style={{
        background: '#FFFFFF',
        border: '1px solid #E2E8F0',
        borderRadius: 8,
        padding: '8px 12px',
        fontSize: '0.8125rem',
        boxShadow: '0 4px 6px -1px rgba(15, 23, 42, 0.08)',
        color: '#111827'
      }}>
        <div style={{ fontWeight: 700, marginBottom: 2 }}>{d.payload.name}</div>
        <div style={{
          color: isLocal ? (d.value >= 0 ? '#10B981' : '#EF4444') : '#2559F4',
          fontWeight: 600
        }}>
          {isLocal
            ? `SHAP Impact: ${d.value >= 0 ? '+' : ''}${d.value?.toFixed(4)}`
            : `Mean |SHAP|: ${d.value?.toFixed(4)}`}
        </div>
        {d.payload.significance && (
          <div style={{ color: '#64748B', marginTop: 2, fontSize: '0.75rem' }}>
            Significance: <strong>{d.payload.significance}</strong>
          </div>
        )}
      </div>
    )
  }

  return (
    <>
      {/* Local SHAP contributions */}
      {localData.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon" style={{ background: '#F5F3FF', color: '#7C3AED' }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                  <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                  <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
              </div>
              <div>
                <h3 className="card-title">Key Decision Drivers (Local SHAP)</h3>
                <p className="card-subtitle">Quantified impact of applicant attributes on approval odds</p>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 16, marginBottom: 12, fontSize: '0.75rem', fontWeight: 600 }}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: '#065F46' }}>
              <span style={{ width: 10, height: 10, borderRadius: 2, background: '#10B981' }} />
              Positively Influences Approval
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, color: '#991B1B' }}>
              <span style={{ width: 10, height: 10, borderRadius: 2, background: '#EF4444' }} />
              Increases Credit Risk
            </span>
          </div>

          <div style={{ width: '100%', height: Math.max(260, localData.length * 28) }}>
            <ResponsiveContainer>
              <BarChart data={localData} layout="vertical" margin={{ top: 5, right: 25, left: 130, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#64748B', fontSize: 11 }} />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={125}
                  tick={{ fill: '#334155', fontSize: 11, fontWeight: 500 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {localData.map((entry, i) => (
                    <Cell key={i} fill={entry.value >= 0 ? '#10B981' : '#EF4444'} fillOpacity={0.9} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Natural-language explanation */}
      {natural_language_explanation && (
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon" style={{ background: '#FFFBEB', color: '#D97706' }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                  <line x1="16" y1="13" x2="8" y2="13"></line>
                  <line x1="16" y1="17" x2="8" y2="17"></line>
                  <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
              </div>
              <div>
                <h3 className="card-title">Decision Rationale & Compliance Notice</h3>
                <p className="card-subtitle">Automated adverse action & explanatory summary</p>
              </div>
            </div>
          </div>
          <div className="nl-card-content">{natural_language_explanation}</div>
        </div>
      )}

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon" style={{ background: '#ECFDF5', color: '#059669' }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 1 1 7.072 0l-.548.547A3.374 3.374 0 0 0 14 18.469V19a2 2 0 1 1-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                </svg>
              </div>
              <div>
                <h3 className="card-title">Actionable Profile Recommendations</h3>
                <p className="card-subtitle">Prescribed measures to improve credit profile eligibility</p>
              </div>
            </div>
          </div>

          <div className="rec-list">
            {recommendations.map((r, i) => {
              const impactClass = r.impact ? `${r.impact.toLowerCase().replace(' ', '-')}-impact` : ''
              return (
                <div className="rec-card" key={i}>
                  <div className="rec-header">
                    <span className="rec-title-text">{r.title}</span>
                    {r.impact && (
                      <span className={`rec-badge ${impactClass}`}>
                        {r.impact}
                      </span>
                    )}
                  </div>
                  <div className="rec-detail">{r.detail}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Global importance */}
      {globalData.length > 0 && (
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon" style={{ background: '#EEF2FF', color: '#2559F4' }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="2" y1="12" x2="22" y2="12"></line>
                  <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                </svg>
              </div>
              <div>
                <h3 className="card-title">Global Model Drivers (Mean |SHAP|)</h3>
                <p className="card-subtitle">Overall feature weighting across the training population</p>
              </div>
            </div>
          </div>
          <div style={{ width: '100%', height: Math.max(220, globalData.length * 26) }}>
            <ResponsiveContainer>
              <BarChart data={globalData} layout="vertical" margin={{ top: 5, right: 25, left: 130, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#64748B', fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={125} tick={{ fill: '#334155', fontSize: 11, fontWeight: 500 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} fill="#2559F4" fillOpacity={0.85} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </>
  )
}
