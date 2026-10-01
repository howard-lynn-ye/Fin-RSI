# Market data (raw vendor-style feed)

`quotes.csv`: daily open, high, low, close and volume for 12 US-listed instruments, as
quoted on each day. Quotes are NOT adjusted for splits or dividends.
`corporate_actions.csv`: dated splits (value = new shares per old share) and cash dividends
(value = cash per share on the ex-date). A split changes the quote basis on its date, so
returns computed across a split date from raw quotes are wrong unless adjusted.
Dates are exchange sessions (US). All files end at the current decision date.
