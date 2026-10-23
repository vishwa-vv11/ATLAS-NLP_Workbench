export function Matrix({
  table,
  columns,
  decimals = false,
}: {
  table: number[][];
  columns: string[];
  decimals?: boolean;
}) {
  return (
    <div className="matrix-scroll">
      <table className="data-matrix">
        <thead>
          <tr>
            <th>DOC</th>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.map((row, rowIndex) => (
            <tr key={rowIndex}>
              <td>D{String(rowIndex + 1).padStart(2, "0")}</td>
              {row.map((value, columnIndex) => (
                <td
                  className={value > 0 ? "matrix-active" : ""}
                  key={`${rowIndex}-${columnIndex}`}
                >
                  {decimals ? value.toFixed(2) : value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
