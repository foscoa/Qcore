import win32com.client
from datetime import datetime

# Define the email body as HTML
html_body = """
<html>
<head>
    <style>
        body {{ font-family: Calibri, sans-serif; font-size: 14px; }}
        pre {{ font-family: Consolas, monospace; font-size: 13px; }}
        .section {{ margin-bottom: 15px; }}
    </style>
</head>
<body>
    <p>Dear all,</p>

    <p>Please find the daily report below.</p>

    <div class="section">
        <b>Daily PnL (EUR):</b><br>
        <pre>-   5,609       |   - 0.08%</pre>
    </div>

    <div class="section">
        <b>MTD PnL (EUR):</b><br>
        <pre>+   4,001.93    |   + 0.06%</pre>
    </div>

    <div class="section">
        <b>YTD PnL (EUR):</b><br>
        <pre>-  67,471.39    |   - 1.75%</pre>
    </div>

    <div class="section">
        <b>Monthly Breakdown:</b><br>
        <pre>Mar: +93,702.30      |   + 0.81%</pre>
        <pre>Feb: -118,913.80     |   - 1.63%</pre>
        <pre>Jan: -71,475.02      |   - 0.98%</pre>
    </div>

    <div class="section">
        <b>Report generated on:</b><br>
        <pre>Thursday, 24 April 2025 - 17:22:03 CEST</pre>
    </div>

    <div class="section">
        <b>Fund Exposure:</b><br>
        <pre>Open Positions:             8</pre>
        <pre>Working Positions:          7</pre>
        <pre>Net Liquidation Value:      6'669'889 EUR</pre>
        <pre>Total Risk (all stops hit): 72'225 EUR | 108.28 bps</pre>
        <pre>Gross Exposure (Trading):   4'295'839 EUR | 64.41%</pre>
    </div>
</body>
</html>
"""

# Create the email
outlook = win32com.client.Dispatch("Outlook.Application")
mail = outlook.CreateItem(0)  # 0: Mail item

mail.Subject = "Daily Fund Report - 24 April 2025"
mail.HTMLBody = html_body

# Display the email (you can use mail.Send() to send it directly)
mail.Display()
