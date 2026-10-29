import { useState, useEffect } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

export default function ModelComparison({ apiBase }) {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    axios.get(`${apiBase}/api/metrics`)
      .then(r => setMetrics(r.data))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [apiBase])

  if (loading) {
    return (
      <div className="card loading-card">
        <div className="spinner" />
        <h3 className="loading-title">Loading Evaluation Metrics</h3>
        <p className="loading-desc">Retrieving cross-validation benchmarks and confusion matrices...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="error-card">
        <div className="error-icon">&#9888;</div>
        <div className="error-content">
          <div className="error-title">Failed to load model metrics</div>
          <div className="error-msg">{error}</div>
        </div>
      </div>
    )
  }

  if (!metrics) return null

  const bestModel = metrics.best_model
  const modelNames = ['Logistic Regression', 'Random Forest', 'Decision Tree']
  const activeMetrics = metrics[bestModel]

  // Model algorithms descriptions
  const ALGORITHMS = {
    'Logistic Regression': 'Generalized Linear Model (L2 Regularized)',
    'Random Forest': 'Ensemble Bagging (100 Estimators)',
    'Decision Tree': 'Recursive Partitioning (CART)',
  }

  // Comparison chart data
  const chartData = modelNames.filter(n => metrics[n]).map(n => ({
    name: n,
    shortName: n.replace('Logistic Regression', 'LogReg').replace('Random Forest', 'RandForest').replace('Decision Tree', 'DecTree'),
    accuracy: +(metrics[n].accuracy * 100).toFixed(2),
    f1: +(metrics[n].f1 * 100).toFixed(2),
    roc_auc: +(metrics[n].roc_auc * 100).toFixed(2),
  }))

  const ablation = metrics['RF (no cibil_score) ablation']

  const CustomChartTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null
    return (
      <div style={{
        background: '#FFFFFF',
        border: '1px solid #E2E8F0',
        borderRadius: 8,
        padding: '10px 14px',
        fontSize: '0.8125rem',
        boxShadow: '0 4px 6px -1px rgba(15, 23, 42, 0.08)',
        color: '#111827'
      }}>
        <div style={{ fontWeight: 700, marginBottom: 6 }}>{label}</div>
        {payload.map((item, idx) => (
          <div key={idx} style={{ color: item.color, fontWeight: 600, margin: '2px 0' }}>
            {item.name}: {item.value}%
          </div>
        ))}
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Active Production Model Hero Banner */}
      {activeMetrics && (
        <div className="active-model-banner">
          <div className="active-model-left">
            <div className="active-model-tag">
              <span className="status-dot" style={{ background: '#38BDF8' }} />
              Active Production Champion
            </div>
            <div className="active-model-title">{bestModel} Classifier</div>
            <div className="active-model-meta">
              {ALGORITHMS[bestModel] || 'Supervised Classification'} &bull; Artifact verified & calibrated
            </div>
          </div>

          <div className="active-model-kpis">
            <div className="active-kpi-item">
              <div className="active-kpi-label">Test Accuracy</div>
              <div className="active-kpi-val">{(activeMetrics.accuracy * 100).toFixed(2)}%</div>
            </div>
            <div className="active-kpi-item">
              <div className="active-kpi-label">F1-Score</div>
              <div className="active-kpi-val">{(activeMetrics.f1 * 100).toFixed(2)}%</div>
            </div>
            <div className="active-kpi-item">
              <div className="active-kpi-label">ROC-AUC</div>
              <div className="active-kpi-val">{(activeMetrics.roc_auc * 100).toFixed(2)}%</div>
            </div>
            {metrics.avg_latency_ms_per_sample != null && (
              <div className="active-kpi-item">
                <div className="active-kpi-label">Latency</div>
                <div className="active-kpi-val" style={{ color: '#A7F3D0' }}>
                  {metrics.avg_latency_ms_per_sample} ms
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Model Benchmark Comparison Chart */}
      <div className="card">
        <div className="card-header">
          <div className="card-title-group">
            <div className="card-icon" style={{ background: '#EEF2FF', color: '#2559F4' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
              </svg>
            </div>
            <div>
              <h3 className="card-title">Model Performance Benchmarks</h3>
              <p className="card-subtitle">Comparison across holdout test set (854 unseen loan applications)</p>
            </div>
          </div>
        </div>

        <div style={{ width: '100%', height: 310 }}>
          <ResponsiveContainer>
            <BarChart data={chartData} margin={{ top: 15, right: 25, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: '#334155', fontSize: 12, fontWeight: 600 }} />
              <YAxis domain={[80, 100]} tick={{ fill: '#64748B', fontSize: 11 }} unit="%" />
              <Tooltip content={<CustomChartTooltip />} />
              <Legend wrapperStyle={{ paddingTop: 10, fontSize: '0.8125rem' }} />
              <Bar dataKey="accuracy" name="Accuracy %" fill="#2559F4" radius={[4, 4, 0, 0]} />
              <Bar dataKey="f1" name="F1-Score %" fill="#10B981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="roc_auc" name="ROC-AUC %" fill="#0284C7" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Comprehensive Metrics Table */}
      <div className="card">
        <div className="card-header">
          <div className="card-title-group">
            <div className="card-icon" style={{ background: '#F8FAFC', color: '#334155' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                <line x1="3" y1="9" x2="21" y2="9"></line>
                <line x1="3" y1="15" x2="21" y2="15"></line>
                <line x1="9" y1="3" x2="9" y2="21"></line>
              </svg>
            </div>
            <div>
              <h3 className="card-title">Comprehensive Model Evaluation Metrics</h3>
              <p className="card-subtitle">Detailed test scores and 5-fold cross-validation estimates</p>
            </div>
          </div>
        </div>

        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Candidate Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>ROC-AUC</th>
                <th>5-Fold CV Accuracy</th>
                <th>5-Fold CV F1</th>
              </tr>
            </thead>
            <tbody>
              {modelNames.filter(n => metrics[n]).map(name => {
                const m = metrics[name]
                const isBest = name === bestModel
                return (
                  <tr key={name} className={isBest ? 'best-model-row' : ''}>
                    <td>
                      {name}
                      {isBest && <span className="active-badge">Selected</span>}
                    </td>
                    <td>{(m.accuracy * 100).toFixed(2)}%</td>
                    <td>{(m.precision * 100).toFixed(2)}%</td>
                    <td>{(m.recall * 100).toFixed(2)}%</td>
                    <td>{(m.f1 * 100).toFixed(2)}%</td>
                    <td>{(m.roc_auc * 100).toFixed(2)}%</td>
                    <td>
                      {m.cv_accuracy_mean != null
                        ? `${(m.cv_accuracy_mean * 100).toFixed(2)}% (±${(m.cv_accuracy_std * 100).toFixed(2)})`
                        : '-'}
                    </td>
                    <td>
                      {m.cv_f1_mean != null
                        ? `${(m.cv_f1_mean * 100).toFixed(2)}% (±${(m.cv_f1_std * 100).toFixed(2)})`
                        : '-'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Confusion Matrix Inspection Grid */}
      <div className="card">
        <div className="card-header">
          <div className="card-title-group">
            <div className="card-icon" style={{ background: '#ECFDF5', color: '#059669' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="7" height="7"></rect>
                <rect x="14" y="3" width="7" height="7"></rect>
                <rect x="14" y="14" width="7" height="7"></rect>
                <rect x="3" y="14" width="7" height="7"></rect>
              </svg>
            </div>
            <div>
              <h3 className="card-title">Confusion Matrices (Holdout Set)</h3>
              <p className="card-subtitle">Actual counts of True Negatives, False Positives, False Negatives, and True Positives</p>
            </div>
          </div>
        </div>

        <div className="cm-grid-container">
          {modelNames.filter(n => metrics[n]?.confusion_matrix).map(name => {
            const cm = metrics[name].confusion_matrix
            const tn = cm[0][0]
            const fp = cm[0][1]
            const fn = cm[1][0]
            const tp = cm[1][1]
            const total = tn + fp + fn + tp
            const isBest = name === bestModel

            return (
              <div className="cm-card" key={name} style={isBest ? { borderColor: '#2559F4', background: '#F8FAFF' } : {}}>
                <div className="cm-card-header">
                  <span>{name}</span>
                  {isBest && <span className="active-badge">Active Model</span>}
                </div>

                <table className="cm-table">
                  <tbody>
                    <tr>
                      <td className="cm-cell hit" title="True Negative (Correct Rejection)">
                        <div className="cm-cell-count">{tn}</div>
                        <div className="cm-cell-label">True Negative</div>
                      </td>
                      <td className="cm-cell miss" title="False Positive (Type I Error)">
                        <div className="cm-cell-count">{fp}</div>
                        <div className="cm-cell-label">False Positive</div>
                      </td>
                    </tr>
                    <tr>
                      <td className="cm-cell miss" title="False Negative (Type II Error)">
                        <div className="cm-cell-count">{fn}</div>
                        <div className="cm-cell-label">False Negative</div>
                      </td>
                      <td className="cm-cell hit" title="True Positive (Correct Approval)">
                        <div className="cm-cell-count">{tp}</div>
                        <div className="cm-cell-label">True Positive</div>
                      </td>
                    </tr>
                  </tbody>
                </table>
                <div style={{ fontSize: '0.725rem', color: '#64748B', marginTop: 10, textAlign: 'center' }}>
                  Total Evaluated: {total} applicants &bull; Accuracy: {((tn + tp) / total * 100).toFixed(2)}%
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Feature Importance per Model */}
      <div className="card">
        <div className="card-header">
          <div className="card-title-group">
            <div className="card-icon" style={{ background: '#F5F3FF', color: '#7C3AED' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
              </svg>
            </div>
            <div>
              <h3 className="card-title">Top-5 Intrinsic Feature Weights</h3>
              <p className="card-subtitle">Key variables prioritized by each algorithm</p>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 20 }}>
          {modelNames.filter(n => metrics[n]?.top5_importance?.length).map(name => {
            const data = metrics[name].top5_importance.map(f => ({
              name: f.feature,
              importance: +(f.importance * 100).toFixed(2),
            }))

            return (
              <div key={name} style={{ background: '#F8FAFC', border: '1px solid #E2E8F0', borderRadius: 10, padding: '14px 16px' }}>
                <div style={{ fontSize: '0.875rem', fontWeight: 700, color: '#111827', marginBottom: 10 }}>
                  {name}
                </div>
                <div style={{ width: '100%', height: 180 }}>
                  <ResponsiveContainer>
                    <BarChart data={data} layout="vertical" margin={{ top: 0, right: 20, left: 80, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#EDF2F7" horizontal={false} />
                      <XAxis type="number" tick={{ fill: '#64748B', fontSize: 10 }} unit="%" />
                      <YAxis type="category" dataKey="name" width={80} tick={{ fill: '#334155', fontSize: 10, fontWeight: 500 }} />
                      <Tooltip
                        contentStyle={{
                          background: '#FFFFFF',
                          border: '1px solid #E2E8F0',
                          borderRadius: 6,
                          fontSize: '0.75rem',
                          color: '#111827'
                        }}
                      />
                      <Bar dataKey="importance" fill="#2559F4" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Ablation Study */}
      {ablation && (
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon" style={{ background: '#FEF3C7', color: '#B45309' }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                  <line x1="12" y1="9" x2="12" y2="13"></line>
                  <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
              </div>
              <div>
                <h3 className="card-title">Ablation Study: Model Reliance on Credit Score</h3>
                <p className="card-subtitle">Random Forest retrained without <code>cibil_score</code> to test dependence on credit bureau history</p>
              </div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
            <div className="metric-box">
              <div className="metric-box-label">Ablated Accuracy</div>
              <div className="metric-box-value">{(ablation.accuracy * 100).toFixed(2)}%</div>
              <div style={{ fontSize: '0.725rem', color: '#EF4444', marginTop: 2 }}>
                Down from {(activeMetrics?.accuracy * 100).toFixed(2)}% (-{((activeMetrics?.accuracy - ablation.accuracy) * 100).toFixed(2)}%)
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-box-label">Ablated F1-Score</div>
              <div className="metric-box-value">{(ablation.f1 * 100).toFixed(2)}%</div>
              <div style={{ fontSize: '0.725rem', color: '#EF4444', marginTop: 2 }}>
                Down from {(activeMetrics?.f1 * 100).toFixed(2)}% (-{((activeMetrics?.f1 - ablation.f1) * 100).toFixed(2)}%)
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-box-label">Ablated ROC-AUC</div>
              <div className="metric-box-value">{(ablation.roc_auc * 100).toFixed(2)}%</div>
              <div style={{ fontSize: '0.725rem', color: '#EF4444', marginTop: 2 }}>
                Down from {(activeMetrics?.roc_auc * 100).toFixed(2)}% (-{((activeMetrics?.roc_auc - ablation.roc_auc) * 100).toFixed(2)}%)
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
