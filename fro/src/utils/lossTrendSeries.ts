export const lossTrendSeries = [
  { key: 'Lid', legacyKey: 'Lfeat', formula: 'L_{\\mathrm{id}}', name: '声音身份目标差距', color: '#0891b2' },
  { key: 'Lsem', formula: 'L_{\\mathrm{sem}}', name: '表达内容目标差距', color: '#16a34a' },
  { key: 'Lpsy', formula: 'L_{\\mathrm{psy}}', name: '听感保真目标差距', color: '#d97706' },
  { key: 'L2', formula: 'L_2', name: '扰动幅度目标差距', color: '#8b5cf6' },
  { key: 'total', formula: 'L_{\\mathrm{total}}', name: '综合目标差距', color: '#e11d48' },
] as const
