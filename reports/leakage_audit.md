\# Leakage and column audit



Columns removed before modeling, and why.



| Column(s) | Why removed |

|---|---|

| out\_prncp, out\_prncp\_inv, total\_pymnt, total\_pymnt\_inv, total\_rec\_prncp, total\_rec\_int, total\_rec\_late\_fee, recoveries, collection\_recovery\_fee | ... |

| last\_pymnt\_d, last\_pymnt\_amnt, next\_pymnt\_d, last\_credit\_pull\_d, last\_fico\_range\_high, last\_fico\_range\_low | ... |

| debt\_settlement\_flag, pymnt\_plan, policy\_code | ... |

| loan\_status | ... (it is the source of the target `default`) |

| id, member\_id, url, desc, title, emp\_title, zip\_code | ... |

| funded\_amnt, funded\_amnt\_inv | ... |

| hardship\_\*, settlement\_\*, sec\_app\_\*, orig\_projected\*, deferral\* | ... |

| issue\_d, earliest\_cr\_line, term, emp\_length | ... (converted into other columns: credit\_hist\_years, term\_months, emp\_length\_yrs) |

| 23 columns with more than 50% missing values in the training split | ... |



\## Kept on purpose



grade, sub\_grade and int\_rate are assigned by Lending Club at issuance, so they are legitimate inputs, but they are very predictive. ...

