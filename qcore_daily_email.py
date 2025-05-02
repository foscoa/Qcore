import win32com.client
from datetime import datetime

import win32com.client as win32
import pandas as pd

import win32com.client as win32

daily_PnL = +19032  # EUR, from IBKR

est_MTD = -0.27     # previous day, % from Paul's daily estimate
est_YTD = -2.16     # previous day, % from Paul's daily estimate

today = datetime.today().strftime('%d %B %Y')

file_path_open_risks = "Q_Pareto_Transaction_History_DEV/Data/open_risks.csv"

# Sample DataFrames
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

    html = """
       <table>
       <tr><th></th><th>EUR</th><th>Change (%)</th></tr>
       <tr>
           <td>Daily PnL</td>
           <td style="color:{color}">{daily_PnL:,.0f}</td>
           <td style="color:{color}">{ret}</td>
       </tr>
       </table>
       """.format(color=color, daily_PnL=daily_PnL, ret=ret)

    return html

def generate_monthly_returns_table_horizontal(curr_MTD, curr_YTD):
    returns = {
        "Jan": "-0.98%",
        "Feb": "-1.56%",
        "Mar": "+0.81%",
        "Apr": "-0.16%*",
        "May": curr_MTD + "*",
        "Jun": "",
        "Jul": "",
        "Aug": "",
        "Sep": "",
        "Oct": "",
        "Nov": "",
        "Dec": "",
        "YTD": curr_YTD + "*"
    }

    # Start HTML table
    html = """
    <h3>Monthly Returns 2025 - Q Pareto Trading Fund SP, Share Class A – Series 1</h3>
    <table>
    <tr>
    """

    # First row: Months
    for month in returns.keys():
        html += f"<th>{month}</th>"
    html += "</tr><tr>"

    # Second row: Returns
    for ret in returns.values():
        color = "darkgreen" if "+" in ret else "darkred" if "-" in ret else "black"
        html += f"<td style='color:{color}'>{ret}</td>"
    html += "</tr></table>"

    return html

def generate_fund_exposure_table(df):

    def get_number_positions(df):
        nr_positions = df["Status"].value_counts().to_dict()
        if 'working' not in nr_positions.keys():
            nr_positions['working'] = 0

        if 'open' not in nr_positions.keys():
            nr_positions['open'] = 0
        return nr_positions

    NLV = df['NLV'].unique()[0]

    ("{:,.0f}".format(
        sample_data.loc[df["Status"] == "open", "Risk (EUR)"].sum()).replace(",", "'") +
     " EUR | " + str(
        round(sample_data.loc[df["Status"] == "open", "Risk (EUR)"].sum() / NLV * 10000,
              2)) + " bps")

    html = ("""
    <h3>Fund Exposure</h3>
    <table>
        <tr><td><b>Open Positions</b></td><td> """ + str(get_number_positions(df)['open']) + """
    </td></tr>
        <tr><td><b>Working Positions</b></td><td>"""+ str(get_number_positions(df)['working']) +"""</td></tr>
        <tr><td><b>Net Liquidation Value (NLV)</b></td><td>""" + f"{int(NLV):,}" + """ EUR</td></tr>
        <tr><td><b>Total Risk (all stops hit)</b></td><td>""" +

            "{:,.0f}".format(
                sample_data.loc[df["Status"] == "open", "Risk (EUR)"].sum()).replace(",", "'") +
            " EUR | " + str(
                round(sample_data.loc[df["Status"] == "open", "Risk (EUR)"].sum() / NLV * 10000,
                      2)) + " bps"

        + """</td></tr>
        <tr><td><b>Gross Exposure (Trading)</b></td><td>""" +

            "{:,.0f}".format(
                sample_data.loc[df["Status"] == "open", "Exposure (EUR)"].sum()).replace(",", "'") +
            " EUR | " + str(
                round(sample_data.loc[df["Status"] == "open", "Exposure (EUR)"].sum() / NLV * 100,
                      2)) + "%"

            + """</td></tr>
    </table>
    """)
    return html

def generate_positions_html(df, type):
    # Filter open positions
    positions = df[df["Status"] == type]
    
    # Check if there are no open positions
    if positions.empty:
        return f"""
        <h3>{type.capitalize()} Positions</h3>
        None
    """ # Or any other message you prefer

    # Initialize an empty string to store the HTML list items
    positions_html = "<ul>"  # Start an unordered list

    # Iterate through the open positions and build the HTML list item
    for _, row in positions.iterrows():
        # Extract relevant data for each open position
        position_dir = row.get("Position", "N/A")  # Assuming 'Position' is a column
        name = row.get("Name", "N/A")  # Assuming 'Name' is a column

        # Determine the color for LONG or SHORT
        if position_dir == "LONG":
            color = "darkgreen"
        elif position_dir == "SHORT":
            color = "darkred"
        else:
            color = "black"  # Default color if not LONG or SHORT

        # Add the HTML for each open position with italicized 'Position'
        positions_html += f"""
        <li>
            <span style="color: {color};">{position_dir}</span> - <i>{name}</i>
        </li>
        """

    positions_html += "</ul>"  # Close the unordered list

    # Final HTML string including the open positions
    html = f"""
        <h3>{type.capitalize()} Positions</h3>
        {positions_html}
    """

    return html


# FULL BODY #

def generate_full_email_body():
    html = """
    <html>
    <head>
    <style>
        body { font-family: Aptos, sans-serif; font-size: 11pt; }
        table { border-collapse: collapse; width: 95%; margin-bottom: 0px; }
        th, td { border: 1px solid #dddddd; text-align: center; padding: 4px; width: 7.5%}
        th { background-color: #f2f2f2; }
        h3 { color: #12365a; }
    </style>
    </head>
    <body>
    <p>Dear all,<br><br>please find below today's report:<br></p>
    """

    html += generate_pnl_table(daily_PnL)
    html += generate_monthly_returns_table_horizontal(curr_MTD, curr_YTD)
    html += """
        <p><i>*estimated on """ + report_time + """</i></p>
        </body></html>
        """
    html += generate_fund_exposure_table(df=sample_data)

    html += generate_positions_html(df=sample_data, type='open')
    html += generate_positions_html(df=sample_data, type='working')
    html += generate_positions_html(df=sample_data, type='closed')

    # Closing the email
    html += """
            <p>Best regards,<br>Team Q - PT</p>
        </body>
        </html>
        """

    return html

def send_outlook_email(subject, html_body, recipients):
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)

    mail.Subject = subject
    mail.To = recipients  # Example: "example@example.com;another@example.com"

    # Set the sender's email (optional, defaults to your Outlook account)
    mail.Sender = "fa@qcore.group"  # Optional: Can be used when sending on behalf of someone else

    mail.BodyFormat = 2  # 2 = HTML
    mail.HTMLBody = html_body
    # Display the email (you can use mail.Send() to send it directly)
    mail.Display()


    # mail.Send()  # .Display() if you want to preview instead of sending directly

if __name__ == "__main__":
    body = generate_full_email_body()

    subject = "📈 Q-PT Report - " + today + " | Daily PnL: " + f"{daily_PnL/NLV:+.2%}" + "*"
    recipients = "fosco.antognini@qcore.ch"  # <--- put real emails separated by ;

    send_outlook_email(subject, body, recipients)
    print("Email sent successfully!")


