import smtplib
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

#### TO MODIFY #########################################################################################################

daily_PnL = 256  # EUR, from IBKR
est_MTD = +0.14    # previous day, % from Paul's daily estimate
est_YTD = -0.07     # previous day, % from Paul's daily estimate

########################################################################################################################

today = datetime.today().strftime('%d %B %Y')

file_path_open_risks = "Q_Pareto_Transaction_History_DEV/Data/open_risks.csv"

def get_sample_data(file_path):
    return pd.read_csv(file_path, index_col=0)

sample_data = get_sample_data(file_path_open_risks)
report_time = sample_data['Report Time'].unique()[0]
NLV = sample_data['NLV'].unique()[0]

curr_MTD = est_MTD/100 + daily_PnL/NLV
curr_MTD = f"{curr_MTD:+.2%}"

curr_YTD = est_YTD/100 + daily_PnL/NLV
curr_YTD = f"{curr_YTD:+.2%}"

def generate_pnl_table(daily_PnL):
    ret = f"{daily_PnL / NLV:+.2%}" + "*"
    color = "darkgreen" if daily_PnL > 0 else "darkred" if daily_PnL < 0 else "black"

    html = f"""
       <table>
       <tr><th></th><th>EUR</th><th>Change (%)</th></tr>
       <tr>
           <td>Daily PnL</td>
           <td style="color:{color}">{daily_PnL:,.0f}</td>
           <td style="color:{color}">{ret}</td>
       </tr>
       </table>
       """
    return html

def generate_monthly_returns_table_horizontal(curr_MTD, curr_YTD):
    returns = {
        "Jan": "-0.98%",
        "Feb": "-1.56%",
        "Mar": "+0.81%",
        "Apr": "-0.24%",
        "May": "+1.79%*",
        "Jun": curr_MTD + "*",
        "Jul": "",
        "Aug": "",
        "Sep": "",
        "Oct": "",
        "Nov": "",
        "Dec": "",
        "YTD": curr_YTD + "*"
    }

    html = """
    <h3>Monthly Returns 2025 - Q Pareto Trading Fund SP, Share Class A – Series 1</h3>
    <table>
    <tr>
    """
    for month in returns.keys():
        html += f"<th>{month}</th>"
    html += "</tr><tr>"
    for ret in returns.values():
        color = "darkgreen" if "+" in ret else "darkred" if "-" in ret else "black"
        html += f"<td style='color:{color}'>{ret}</td>"
    html += "</tr></table>"

    return html

def generate_fund_exposure_table(df):
    def get_number_positions(df):
        counts = df["Status"].value_counts().to_dict()
        return {'open': counts.get('open', 0), 'working': counts.get('working', 0)}

    open_risk = df[df["Status"] == "open"]["Risk (EUR)"].sum()
    open_exposure = df[df["Status"] == "open"]["Exposure (EUR)"].sum()

    html = f"""
    <h3>Fund Exposure</h3>
    <table>
        <tr><td><b>Open Positions</b></td><td>{get_number_positions(df)['open']}</td></tr>
        <tr><td><b>Working Positions</b></td><td>{get_number_positions(df)['working']}</td></tr>
        <tr><td><b>Net Liquidation Value (NLV)</b></td><td>{int(NLV):,} EUR</td></tr>
        <tr><td><b>Total Risk (all stops hit)</b></td><td>{open_risk:,.0f} EUR | {round(open_risk / NLV * 10000, 2)} bps</td></tr>
        <tr><td><b>Gross Exposure (Trading)</b></td><td>{open_exposure:,.0f} EUR | {round(open_exposure / NLV * 100, 2)}%</td></tr>
    </table>
    """
    return html

def generate_positions_html(df, type):
    positions = df[df["Status"] == type]
    if positions.empty:
        return f"<h3>{type.capitalize()} Positions</h3>None"

    html = f"<h3>{type.capitalize()} Positions</h3><ul>"
    for _, row in positions.iterrows():
        direction = row.get("Position", "N/A")
        name = row.get("Name", "N/A")
        color = "darkgreen" if direction == "LONG" else "darkred" if direction == "SHORT" else "black"
        html += f"<li><span style='color:{color}'>{direction}</span> - <i>{name}</i></li>"
    html += "</ul>"
    return html

def generate_full_email_body():
    html = """
    <html>
    <head>
    <style>
        body { font-family: Helvetica, sans-serif; font-size: 11pt; }
        table { border-collapse: collapse; width: 95%; }
        th, td { border: 1px solid #dddddd; text-align: center; padding: 4px; }
        th { background-color: #f2f2f2; }
        h3 { color: #12365a; }
    </style>
    </head>
    <body>
    <p>Dear all,<br><br>please find below today's report:<br></p>
    """

    html += generate_pnl_table(daily_PnL)
    html += generate_monthly_returns_table_horizontal(curr_MTD, curr_YTD)
    html += f"<p><i>*estimated on {report_time}</i></p>"
    html += generate_fund_exposure_table(df=sample_data)
    html += generate_positions_html(df=sample_data, type='open')
    html += generate_positions_html(df=sample_data, type='working')
    html += generate_positions_html(df=sample_data, type='closed')
    html += "<p>Best regards,<br>Team Q - PT</p></body></html>"

    return html

def send_email_smtp(subject, html_body, recipients):
    from_address = "your_email@domain.com"
    password = "your_password"  # Consider using keychain or env var for security
    smtp_server = "smtp.office365.com"
    smtp_port = 587

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = from_address
    msg['To'] = recipients

    part = MIMEText(html_body, 'html')
    msg.attach(part)

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(from_address, password)
        server.sendmail(from_address, recipients.split(";"), msg.as_string())

if __name__ == "__main__":
    html = generate_full_email_body()

    subject = f"📈 Q-PT Report - {today} | Daily PnL: {daily_PnL / NLV:+.2%}*"
    recipients = "rr@qcore.group;jw@qcore.group;sven.schmidt@qcore.group;pc@qcore.group;sunanda.thiyagarajah@qcore.fund;norman.hartmann@qcore.fund"

    send_email_smtp(subject, html, recipients)
    print("Email sent successfully via SMTP!")
