import { formatINR } from './LoanForm'

const TIER_COLORS = {
  'Prime': '#10B981',       // Emerald
  'Near Prime': '#0284C7',  // Sky Blue
  'Subprime': '#F59E0B',    // Amber
  'High Risk': '#EF4444',   // Red
}

const TIER_ANGLES = {
  'High Risk': { start: 180, end: 225 },
  'Subprime': { start: 225, end: 270 },
  'Near Prime': { start: 270, end: 315 },
  'Prime': { start: 315, end: 360 },
}

function SemiGauge({ probability, tier }) {
  const pct = Math.max(0, Math.min(1, probability || 0))
  const needleAngle = 180 + pct * 180
  const cx = 130, cy = 125, r = 95

  const arcPath = (startDeg, endDeg) => {
    const s = (startDeg * Math.PI) / 180
    const e = (endDeg * Math.PI) / 180
    const x1 = cx + r * Math.cos(s), y1 = cy + r * Math.sin(s)
    const x2 = cx + r * Math.cos(e), y2 = cy + r * Math.sin(e)
    const large = endDeg - startDeg > 180 ? 1 : 0
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`
  }

  const nx = cx + (r - 12) * Math.cos((needleAngle * Math.PI) / 180)
  const ny = cy + (r - 12) * Math.sin((needleAngle * Math.PI) / 180)

  return (
    <div className="gauge-container">
      <svg viewBox="0 0 260 145" width="260" height="145" style={{ overflow: 'visible' }}>
        {/* Track arcs */}
        {Object.entries(TIER_ANGLES).map(([name, { start, end }]) => (
          <path
            key={name}
            d={arcPath(start, end)}
            fill="none"
            stroke={TIER_COLORS[name]}
            strokeWidth="16"
            strokeLinecap="butt"
            opacity={tier === name ? 1 : 0.28}
          />
        ))}

        {/* Center pivot hub */}
        <circle cx={cx} cy={cy} r="6" fill="#111827" />

        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={nx}
          y2={ny}
          stroke="#111827"
          strokeWidth="3"
          strokeLinecap="round"
        />

        {/* Min / Max Labels */}
        <text x="18" y="140" fill="#64748B" fontSize="10" fontWeight="600" fontFamily="Inter">0%</text>
        <text x="228" y="140" fill="#64748B" fontSize="10" fontWeight="600" fontFamily="Inter">100%</text>

        {/* Value text in center */}
        <text
          x={cx}
          y={cy - 12}
          fill="#111827"
          fontSize="24"
          fontWeight="800"
          textAnchor="middle"
          fontFamily="Inter"
        >
          {(pct * 100).toFixed(1)}%
        </text>
      </svg>
      <div className="gauge-caption">
        Estimated Approval Probability
      </div>
    </div>
  )
}

export default function PredictionResult({ result, onNewAssessment }) {
  if (!result) return null

  const { decision, probability, risk_tier, manual_review, engineered_features, model_name } = result
  const displayDecision = manual_review ? 'Manual Review' : decision
  const statusClass = manual_review ? 'manual-review' : decision.toLowerCase()

  return (
    <>
      {/* Primary Decision Card */}
      <div className={`decision-card ${statusClass}`}>
        <div className="decision-header">
          <div className="decision-badge-container">
            <div className={`decision-pill ${statusClass}`}>
              {statusClass === 'approved' && (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
              )}
              {statusClass === 'rejected' && (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="6" x2="6" y2="18"></line>
                  <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
              )}
              {statusClass === 'manual-review' && (
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"></circle>
                  <line x1="12" y1="8" x2="12" y2="12"></line>
                  <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
              )}
              <span>{displayDecision}</span>
            </div>

            <span className="risk-tier-pill">
              Risk Tier: <strong>{risk_tier}</strong>
            </span>
          </div>

          {onNewAssessment && (
            <button
              type="button"
              className="new-assessment-btn"
              onClick={onNewAssessment}
              title="Assess another applicant without reloading page"
            >
              + New Assessment
            </button>
          )}
        </div>

        {/* Semi Gauge */}
        <SemiGauge probability={probability} tier={risk_tier} />

        <div className="disclaimer-text">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="16" x2="12" y2="12"></line>
            <line x1="12" y1="8" x2="12.01" y2="8"></line>
          </svg>
          <span>
            Scored by <strong>{model_name || 'Production Random Forest'}</strong> model pipeline. Automated score is advisory; final loan sanctioning adheres to institutional underwriting guidelines.
          </span>
        </div>
      </div>

      {/* Computed Financial Metrics Card */}
      {engineered_features && (
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon" style={{ background: '#F0F9FF', color: '#0284C7' }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                  <line x1="8" y1="21" x2="16" y2="21"></line>
                  <line x1="12" y1="17" x2="12" y2="21"></line>
                </svg>
              </div>
              <div>
                <h3 className="card-title">Computed Financial Metrics</h3>
                <p className="card-subtitle">Derived ratios analyzed during model inference</p>
              </div>
            </div>
          </div>

          <div className="metrics-grid-6">
            <div className="metric-box">
              <div className="metric-box-label">Loan-to-Income</div>
              <div className="metric-box-value">
                {engineered_features.loan_to_income != null ? engineered_features.loan_to_income.toFixed(2) : '-'}
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-box-label">Monthly EMI (est.)</div>
              <div className="metric-box-value" style={{ fontSize: '0.95rem' }}>
                {formatINR(engineered_features.emi)}
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-box-label">Debt-to-Income</div>
              <div className="metric-box-value">
                {engineered_features.debt_income_ratio != null ? engineered_features.debt_income_ratio.toFixed(2) : '-'}
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-box-label">Income / Dep.</div>
              <div className="metric-box-value" style={{ fontSize: '0.95rem' }}>
                {formatINR(engineered_features.income_per_dep)}
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-box-label">Total Assets</div>
              <div className="metric-box-value" style={{ fontSize: '0.95rem' }}>
                {formatINR(engineered_features.total_assets)}
              </div>
            </div>

            <div className="metric-box">
              <div className="metric-box-label">Asset Coverage</div>
              <div className="metric-box-value">
                {engineered_features.asset_coverage != null ? `${engineered_features.asset_coverage.toFixed(2)}x` : '-'}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
