import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "/Users/juniorjumbong/Desktop/personal-website/00_tds/01_PD_credit_scoring/outputs/modalites_credit_scoring";
const outputPath = path.join(outputDir, "tableau_variables_modalites_credit_scoring.xlsx");

const RED = "#B51E18";
const DARK_RED = "#8E1712";
const LIGHT_RED = "#F8E7E5";
const MID_GRAY = "#E7E7E7";
const LIGHT_GRAY = "#F6F6F6";
const TEXT = "#222222";
const BLUE = "#2F99B8";

const modalitesRows = [
  ["person_income", "(3999.999, 45000.0]", 7367, 2547, 0.558308, 4820, 0.288105, 0.661585, 0.178762],
  ["person_income", "(45000.0, 71000.0]", 6857, 1267, 0.277729, 5590, 0.334130, -0.184885, 0.010428],
  ["person_income", "(71000.0, 141500.0]", 7068, 748, 0.163963, 6320, 0.377764, -0.834629, 0.178445],
  ["person_emp_length", "(-0.001, 2.0]", 7676, 2102, 0.460763, 5574, 0.333174, 0.324219, 0.041367],
  ["person_emp_length", "(2.0, 6.0]", 7617, 1472, 0.322665, 6145, 0.367304, -0.129574, 0.005784],
  ["person_emp_length", "(6.0, 14.5]", 5999, 988, 0.216572, 5011, 0.299522, -0.324266, 0.026898],
  ["loan_int_rate", "(5.419, 9.91]", 7203, 748, 0.163963, 6455, 0.385834, -0.855765, 0.189869],
  ["loan_int_rate", "(9.91, 12.21]", 7093, 1241, 0.272030, 5852, 0.349791, -0.251424, 0.019551],
  ["loan_int_rate", "(12.21, 21.825]", 6996, 2573, 0.564007, 4423, 0.264375, 0.757697, 0.227030],
  ["loan_percent_income", "(-0.001, 0.11]", 7839, 908, 0.199036, 6931, 0.414286, -0.733073, 0.157794],
  ["loan_percent_income", "(0.11, 0.2]", 6992, 1043, 0.228628, 5949, 0.355589, -0.441680, 0.056076],
  ["loan_percent_income", "(0.2, 0.44]", 6461, 2611, 0.572337, 3850, 0.230126, 0.911103, 0.311789],
  ["home_ownership_3", "OTHER_RENT", 10682, "", "", "", "", "", ""],
  ["home_ownership_3", "MORTGAGE", 8907, "", "", "", "", "", ""],
  ["home_ownership_3", "OWN", 1703, "", "", "", "", "", ""],
  ["cb_person_default_on_file", "N", 17526, "", "", "", "", "", ""],
  ["cb_person_default_on_file", "Y", 3766, "", "", "", "", "", ""],
];

const psiRows = [
  ["person_income", 0.0010, 0.0184, 0.0113],
  ["person_emp_length", 0.0006, 0.0136, 0.0087],
  ["loan_int_rate", 0.0000, 0.0001, 0.0001],
  ["loan_percent_income", 0.0002, 0.0020, 0.0010],
  ["home_ownership_3", 0.0002, 0.0034, 0.0020],
  ["cb_person_default_on_file", 0.0000, 0.0001, 0.0002],
];

function styleTitle(sheet, range, title, subtitle = "") {
  sheet.getRange(range).merge();
  const titleRange = sheet.getRange(range.split(":")[0]);
  titleRange.values = [[title]];
  titleRange.format = {
    fill: "#FFFFFF",
    font: { bold: true, color: RED, size: 18 },
    horizontalAlignment: "left",
    verticalAlignment: "middle",
  };

  if (subtitle) {
    sheet.getRange("A2:J2").merge();
    sheet.getRange("A2").values = [[subtitle]];
    sheet.getRange("A2").format = {
      font: { color: "#555555", size: 10 },
      horizontalAlignment: "left",
      verticalAlignment: "middle",
    };
  }
  sheet.getRange("A3:D3").merge();
  sheet.getRange("A3").values = [[""]];
  sheet.getRange("A3:D3").format = { fill: RED };
  sheet.getRange("A3:D3").format.rowHeightPx = 6;
}

function setColumnWidths(sheet) {
  const widths = [44, 180, 205, 82, 82, 92, 96, 112, 82, 82];
  widths.forEach((width, i) => {
    sheet.getRangeByIndexes(0, i, 1, 1).format.columnWidthPx = width;
  });
}

function styleTable(sheet, tableRange, headerRange) {
  sheet.getRange(headerRange).format = {
    fill: DARK_RED,
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "middle",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: "#FFFFFF" },
  };
  sheet.getRange(tableRange).format.borders = { preset: "all", style: "thin", color: "#CFCFCF" };
}

const workbook = Workbook.create();
const main = workbook.worksheets.add("Modalites");
main.showGridLines = false;
setColumnWidths(main);
main.getRange("A1:J1").format.rowHeightPx = 34;
main.getRange("A5:J5").format.rowHeightPx = 32;
styleTitle(
  main,
  "A1:J1",
  "PD CREDIT SCORING - TABLE DES MODALITES",
  "Variables, modalités de discrétisation et indicateurs WoE / IV fournis pour la scorecard."
);

const headers = [["#", "Variable", "Modalité", "N", "Events", "% Events", "Non-Events", "% Non-Events", "WoE", "IV"]];
main.getRange("A5:J5").values = headers;
const bodyValues = modalitesRows.map((row, index) => [index + 1, ...row]);
main.getRangeByIndexes(5, 0, bodyValues.length, headers[0].length).values = bodyValues;

styleTable(main, `A5:J${5 + bodyValues.length}`, "A5:J5");
main.getRange(`A6:J${5 + bodyValues.length}`).format = {
  font: { color: TEXT, size: 10 },
  verticalAlignment: "middle",
};
main.getRange(`A6:A${5 + bodyValues.length}`).format = {
  fill: RED,
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
};
main.getRange(`B6:B${5 + bodyValues.length}`).format = {
  font: { bold: true, color: TEXT },
  horizontalAlignment: "left",
  verticalAlignment: "middle",
};
main.getRange(`C6:C${5 + bodyValues.length}`).format = {
  horizontalAlignment: "left",
  verticalAlignment: "middle",
};
main.getRange(`D6:E${5 + bodyValues.length}`).format.numberFormat = "0";
main.getRange(`G6:G${5 + bodyValues.length}`).format.numberFormat = "0";
main.getRange(`F6:F${5 + bodyValues.length}`).format.numberFormat = "0.00%";
main.getRange(`H6:H${5 + bodyValues.length}`).format.numberFormat = "0.00%";
main.getRange(`I6:J${5 + bodyValues.length}`).format.numberFormat = "0.000000";
main.getRange(`D6:J${5 + bodyValues.length}`).format.horizontalAlignment = "right";

for (let r = 6; r <= 5 + bodyValues.length; r += 1) {
  const fill = r % 2 === 0 ? "#FFFFFF" : LIGHT_GRAY;
  main.getRange(`B${r}:J${r}`).format.fill = fill;
}

for (const startRow of [6, 9, 12, 15, 18, 21]) {
  main.getRange(`A${startRow}:J${startRow}`).format.borders = {
    top: { style: "medium", color: RED },
    insideVertical: { style: "thin", color: "#CFCFCF" },
    bottom: { style: "thin", color: "#CFCFCF" },
    left: { style: "thin", color: "#CFCFCF" },
    right: { style: "thin", color: "#CFCFCF" },
  };
}

main.getRange("L5:N5").values = [["Lecture", "", ""]];
main.getRange("L5:N5").merge();
main.getRange("L5").format = {
  fill: RED,
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
};
main.getRange("L6:N11").values = [
  ["Variables numériques", "modalités par classes", ""],
  ["Variables catégorielles", "modalités observées", ""],
  ["Events / Non-Events", "renseignés si fournis", ""],
  ["WoE / IV", "renseignés si fournis", ""],
  ["PSI", "voir l'onglet PSI", ""],
  ["Source", "données transmises dans le prompt", ""],
];
main.getRange("L6:N11").format = {
  fill: "#FAFAFA",
  font: { color: TEXT, size: 9 },
  borders: { preset: "all", style: "thin", color: "#D9D9D9" },
  wrapText: true,
  verticalAlignment: "middle",
};
main.getRange("L6:L11").format = {
  fill: LIGHT_RED,
  font: { bold: true, color: TEXT, size: 9 },
};
main.getRange("L:N").format.columnWidthPx = 115;
main.freezePanes.freezeRows(5);

styleTable(main, `A5:J${5 + bodyValues.length}`, "A5:J5");
main.getRange(`A6:A${5 + bodyValues.length}`).format = {
  fill: RED,
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
};
for (let r = 6; r <= 5 + bodyValues.length; r += 1) {
  const fill = r % 2 === 0 ? "#FFFFFF" : LIGHT_GRAY;
  main.getRange(`B${r}:J${r}`).format.fill = fill;
}

const psi = workbook.worksheets.add("PSI");
psi.showGridLines = false;
psi.getRange("A1:D1").format.rowHeightPx = 34;
styleTitle(
  psi,
  "A1:D1",
  "PSI - STABILITE DES VARIABLES",
  "Comparaison des distributions Train, Test et OOT pour les variables retenues."
);
psi.getRange("A5:D5").values = [["Variable", "PSI Train vs Test", "PSI Train vs OOT", "PSI Test vs OOT"]];
psi.getRangeByIndexes(5, 0, psiRows.length, 4).values = psiRows;
psi.getRange("A:A").format.columnWidthPx = 210;
psi.getRange("B:D").format.columnWidthPx = 150;
styleTable(psi, `A5:D${5 + psiRows.length}`, "A5:D5");
psi.getRange(`A6:D${5 + psiRows.length}`).format = {
  font: { color: TEXT, size: 10 },
  verticalAlignment: "middle",
  borders: { preset: "all", style: "thin", color: "#CFCFCF" },
};
psi.getRange(`A6:A${5 + psiRows.length}`).format = {
  font: { bold: true, color: TEXT },
  horizontalAlignment: "left",
  verticalAlignment: "middle",
};
psi.getRange(`B6:D${5 + psiRows.length}`).format = {
  numberFormat: "0.00%",
  horizontalAlignment: "right",
  verticalAlignment: "middle",
};
for (let r = 6; r <= 5 + psiRows.length; r += 1) {
  const fill = r % 2 === 0 ? "#FFFFFF" : LIGHT_GRAY;
  psi.getRange(`A${r}:D${r}`).format.fill = fill;
}
psi.getRange("A5:D5").format = {
  fill: BLUE,
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "middle",
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#FFFFFF" },
};
psi.getRange(`A6:D${5 + psiRows.length}`).format.borders = { preset: "all", style: "thin", color: "#CFCFCF" };
for (let r = 6; r <= 5 + psiRows.length; r += 1) {
  const fill = r % 2 === 0 ? "#FFFFFF" : "#DDF1F7";
  psi.getRange(`A${r}:D${r}`).format.fill = fill;
}
psi.freezePanes.freezeRows(5);

const check = await workbook.inspect({
  kind: "table",
  range: "Modalites!A5:J22",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 10,
});
console.log(check.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const mainPreview = await workbook.render({ sheetName: "Modalites", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(path.join(outputDir, "preview_modalites.png"), new Uint8Array(await mainPreview.arrayBuffer()));
const psiPreview = await workbook.render({ sheetName: "PSI", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(path.join(outputDir, "preview_psi.png"), new Uint8Array(await psiPreview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Saved ${outputPath}`);
