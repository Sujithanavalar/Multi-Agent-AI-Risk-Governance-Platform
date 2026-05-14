import { jsPDF } from 'jspdf';
import { Document, Packer, Paragraph, TextRun } from 'docx';
import { saveAs } from 'file-saver';

function rowsToCsv(rows) {
  if (!rows.length) return '';
  const headers = Object.keys(rows[0]);
  const escape = (v) => {
    if (v === null || v === undefined) return '""';
    const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
    return `"${s.replace(/"/g, '""')}"`;
  };
  return [headers.join(','), ...rows.map((row) => headers.map((h) => escape(row[h])).join(','))].join('\n');
}

export function downloadCsv(rows, filename) {
  const blob = new Blob([rowsToCsv(rows)], { type: 'text/csv;charset=utf-8;' });
  saveAs(blob, filename);
}

export function downloadPdfTable(title, rows, filename) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' });
  const margin = 40;
  let y = margin;
  doc.setFontSize(14);
  doc.text(title, margin, y);
  y += 24;
  doc.setFontSize(9);
  if (!rows.length) {
    doc.text('No data.', margin, y);
    doc.save(filename);
    return;
  }
  const headers = Object.keys(rows[0]);
  const lines = rows.slice(0, 80).map((row) =>
    headers.map((h) => `${h}: ${typeof row[h] === 'object' ? JSON.stringify(row[h]) : row[h]}`).join(' | ')
  );
  lines.forEach((line) => {
    const wrapped = doc.splitTextToSize(line, 520);
    if (y > 760) {
      doc.addPage();
      y = margin;
    }
    doc.text(wrapped, margin, y);
    y += wrapped.length * 12 + 6;
  });
  if (rows.length > 80) {
    doc.text(`…and ${rows.length - 80} more rows (export CSV for full data).`, margin, y);
  }
  doc.save(filename);
}

export async function downloadDocxReport(title, rows, filename) {
  const children = [new Paragraph({ children: [new TextRun({ text: title, bold: true, size: 28 })] }), new Paragraph('')];
  rows.slice(0, 200).forEach((row) => {
    const text = Object.entries(row)
      .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`)
      .join(' | ');
    children.push(new Paragraph({ children: [new TextRun({ text, size: 20 })] }));
  });
  if (rows.length > 200) {
    children.push(new Paragraph(`…and ${rows.length - 200} more rows. Export CSV for full data.`));
  }
  const doc = new Document({ sections: [{ children }] });
  const blob = await Packer.toBlob(doc);
  saveAs(blob, filename);
}
