from openai import OpenAI
import yfinance as yf

client = OpenAI(api_key="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")

ticker = "AAPL"

# Fetch the stock information
stock = yf.Ticker(ticker)

# Get the latest news
news = stock.news

count = 0
# Display the news
while count < 4:
    article = news[count]
    print(f"Title: {article['title']} ")
    print(f"Link: {article['link']}")

    url = article['link']

    promt =  (
        f"Forget all your previous instructions. Pretend you are a financial expert. You are "
        f"a financial expert with stock recommendation experience. Answer “YES” if good "
        f"news, “NO” if bad news, or “UNKNOWN” if uncertain in the first line. Then"
        f"elaborate with one short and concise sentence on the next line. Is this headline"
        f"good or bad for the stock price of company name in the short term? Headline: {url}"
    )

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": promt}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

    print("----------------- \n")

    count += 1

###





