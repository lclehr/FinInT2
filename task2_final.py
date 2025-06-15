import numpy as np
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick

###############################################################

# Streamlit style, title etc

###############################################################

#st.set_page_config(layout="wide")
st.title("Monte Carlo - Share Value Simulation")

# Inject CSS overrides
st.markdown("""
    <style>
    html, body, [class*="css"] {
                font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
                font-style: italic !important;
        background-color: #F5F5F5;
        color: #2B2B2B;
    }

    h1, h2, h3, p, span, div, label {
        font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, serif !important;
        font-style: italic !important;
    }

    /* Title */
    h1 {
        color: #E74C3C !important;
    }

    /* Radio buttons styling */
    div.row-widget.stRadio > div {
        gap: 10px;
    }

    div.row-widget.stRadio input[type="radio"] {
        accent-color: #E74C3C !important;
        width: 20px;
        height: 20px;
    }

    div.row-widget.stRadio label {
        font-size: 16px;
    }

    /* Sliders */
    .stSlider > div[data-baseweb="slider"] > div {
        color: #E74C3C !important;
    }

    /* Hide Streamlit footer */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)






###############################################################

#Store data for the two companies

###############################################################

revenue_MSFT = [
    93580,
    91154,
    96571,
    110360,
    125843,
    143015,
    168088,
    198270,
    211915,
    245122
]

revenue_NVDA = [
    5010,
    6910,
    9714,
    11716,
    10918,
    16675,
    26914,
    26974,
    60922,
    130497
]

# EBIT-Werte für MSFT (30.06.x)
msft_ebit = [
    18161,
    26078,
    29025,
    35058,
    42959,
    52959,
    69916,
    83383,
    88523,
    109433
]

# EBIT-Werte für NVDA (Ende Januar x)
nvda_ebit = [
    747,
    1934,
    3210,
    3804,
    2846,
    4532,
    10041,
    4224,
    32972,
    81453
]

msft_roic = [
    10.2488,
    15.7513,
    14.6551,
    8.8075,
    21.4688,
    22.9883,
    28.8729,
    31.7364,
    28.2877,
    28.7142
]

nvda_roic = [
    10.8561,
    23.8825,
    33.6136,
    39.4485,
    21.2026,
    23.1672,
    31.9539,
    12.7490,
    72.3859,
    109.4474
]

#Actual Data from MSFT in Dictionary for easy access
MSFT =  {"Revenue": 245122, 
        "EBIT": 109433,
        "TaxRate": 0.1820,
        "Debt": 97852,
        "Cash": 75543,
        "SharesOut": 7469,
        "Invested Capital": 352524,
        "Sales to Capital": 2,
       "Historic ROIC":msft_roic,
       "Historic Revenue":revenue_MSFT,
       "Historic EBIT": msft_ebit,
        "Ticker": "MSFT",
        "Industry STC": 1.71} #https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/capex.html

#Actual Data fro NVDA in Dictionary for easy access
NVDA = {"Revenue": 130497, 
        "EBIT": 75605,
        "TaxRate": 0.1326,
        "Debt": 10270,
        "Cash": 43210,
        "SharesOut": 24477,
        "Sales to Capital": 2.77,
        "Invested Capital": 80385,
        "Historic ROIC":nvda_roic,
        "Historic Revenue":revenue_NVDA,
        "Historic EBIT": nvda_ebit,
        "Ticker": "NVDA",
        "Industry STC": 1.1} #https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/capex.html


#Iterable Dictionary with company name and var for com
companies = {"Microsoft": MSFT,
             "Nvidia": NVDA}


###############################################################

#Function to Plot the histograms

###############################################################

def plot_histo(array, num_bins=100, title="", price = 0):
    array = np.array(array).flatten()
    frequencies, bin_edges = np.histogram(array, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig, ax = plt.subplots()
    ax.bar(bin_centers, frequencies, width=(bin_edges[1]-bin_edges[0]) * 0.9, color = 'green',alpha=0.8)
    ax.set_xlabel(f'Realisations of {title}')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    if title != "Share Value":
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    else:
        if price != 0:
            ax.axvline(x=price, color='red', linestyle='--', linewidth=2, label=f"Currently: {current_company["Ticker"]}: {price}")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend()
    st.pyplot(fig)

###############################################################

#Function to Plot two histograms in one

###############################################################

def plot_histo_two(array1, array2, num_bins=100, title=""):
    array1 = np.array(array1).flatten()
    array2 = np.array(array2).flatten()
    fig, ax = plt.subplots()
    ax.hist(array1, bins=num_bins, color='green', alpha=0.8, label='Initial Value')
    ax.hist(array2, bins=num_bins, color='red', alpha=0.6, label='Terminal Value')
    ax.set_xlabel(f'Realisations of {title}')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)


###############################################################

#Function to Plot the historical and forecasted points

###############################################################

def plot_quant(values, dataframe, title): #Values should be a pd.Series, indexed 2005 - 2034
    years = list(range(2015,2025+10))
    quantile_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantiles = dataframe.quantile(quantile_levels)
    # Farbverlauf: rot (niedrig) → grün (hoch)
    cmap = plt.cm.get_cmap('RdYlGn')  # nicht "_r"!
    colors = cmap(np.linspace(0.0, 1.0, len(quantile_levels) - 1))
    selected_years = years[10:]
    highlight_count = 10
    fig, ax = plt.subplots()
    bar_colors = ['green'] * (len(values) - highlight_count) + ['gray'] * highlight_count
    ax.bar(years, values, color=bar_colors, alpha=1)
    for i in range(len(quantile_levels) - 1):
        lower = quantiles.loc[quantile_levels[i], selected_years]
        upper = quantiles.loc[quantile_levels[i + 1], selected_years]
    
        if isinstance(lower.iloc[0], (np.ndarray, list)):
            lower = lower.apply(lambda x: x[0])
        if isinstance(upper.iloc[0], (np.ndarray, list)):
            upper = upper.apply(lambda x: x[0])
    
        ax.fill_between(selected_years, lower, upper,
                        color=colors[i], alpha=0.7,
                        label=f'{int(quantile_levels[i]*100)}%-{int(quantile_levels[i+1]*100)}%')
    if all(isinstance(x, (int, float)) for x in years):
        # Nur für numerische Jahreswerte
        xticks = [x for x in years if int(x) % 5 == 0]
        ax.set_xticks(xticks)
    else:
        # Bei Textwerten z.B. "2020E", muss evtl. anders gefiltert werden
        xticks = [x for x in years if x[:4].isdigit() and int(x[:4]) % 5 == 0]
        ax.set_xticks(xticks)
    formatter = FuncFormatter(lambda x, _: int(x))
    if title != "Revenue":
        formatter = FuncFormatter(lambda x, _: f'{x*100:.0f}%')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(f'{title}')
    else:
        ax.set_ylabel(f'{title} in mln USD')
    ax.set_xlabel('Year')
    ax.set_title(f'{title} (with Quantiles)')
    ax.legend(loc='upper left')
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    st.pyplot(fig)
    

###############################################################

#Monte Carlo Simulation 

###############################################################

sample_wacc = []
sample_wacc_t = []
sample_opm = []
sample_rev = []
sample_rev_t = []
value_list = []
sample_stc= []
columns = list(range(2025, 2035)) 
revenue_df = pd.DataFrame(columns=columns)
reinvestment_df = pd.DataFrame(columns=columns)
fcff_df = pd.DataFrame(columns = columns)
ebit_df = pd.DataFrame(columns = columns)
roic_df = pd.DataFrame(columns = columns)
def monte_carlo_EV():
    
    for j in range(int(draws)):
        #Normal Distribution
        drawn_roic_end = np.random.normal(loc=mu_roic, scale=sigma_roic, size=1)[0]
        drawn_wacc = np.random.normal(loc=mu_wacc, scale=sigma_wacc, size=1)[0]
        
        #LogNormal Distribution
        drawn_revenue_growth = np.random.lognormal(mean=mu_revenue_growth, sigma=sigma_revenue_growth, size=1)[0]
        
        min_lim = min(drawn_wacc, drawn_revenue_growth)
        
        #Triangle Distribution
        drawn_op_margin = np.random.triangular(left=l_limit, mode=mode_op_margin, right=u_limit, size=1)[0]
        drawn_stc = np.random.triangular(left=current_company["Industry STC"]-0.1, mode=s_to_c, right=current_company["Industry STC"]+0.1, size=1)[0]
        
        drawn_wacc_end = np.random.triangular(left=0.01, mode=(0.01+max(min_lim, 0.011))/2, right=max(min_lim, 0.011), size=1)[0]
        drawn_revenue_end = np.random.triangular(left=0, mode=drawn_wacc_end/2, right=drawn_wacc_end, size=1)[0]
        
        
        sample_wacc.append(drawn_wacc)
        sample_wacc_t.append(drawn_wacc_end)
        sample_opm.append(drawn_op_margin)
        sample_rev.append(drawn_revenue_growth)
        sample_rev_t.append(drawn_revenue_end)
        sample_stc.append(drawn_stc)
    
        #Revenue Simulation
        revenue_growth_start = drawn_revenue_growth
        revenue_growth_end = drawn_revenue_end
        revenue_subtract = (revenue_growth_start-revenue_growth_end)/5
        revenue = []
        revenue_growth = []
        revenue_growth_amt = []
        for i in range(10):
            if i < 5:
                revenue_growth.append(revenue_growth_start)
            else:
                revenue_growth.append(revenue_growth_start-(i-4)*revenue_subtract)
        revenue_start = current_company["Revenue"]
        for i in range(len(revenue_growth)):
            revenue.append(revenue_start*(1+revenue_growth[i]))
            revenue_start = revenue[i]    
            revenue_growth_amt.append(revenue_start*(1+revenue_growth[i])-revenue_start)
        revenue_df.loc[j] = revenue
        
        #Operating Margin Simulation
        op_margin_input = drawn_op_margin
        op_margin = []
        for i in range(10):
            op_margin.append(op_margin_input)
    
        #turn revenue, op_margin into array for calculations
        revenue = np.array(revenue)
        op_margin = np.array(op_margin)
        revenue_growth_amt = np.array(revenue_growth_amt)
        
        #calculate Ebit
        ebit = op_margin * revenue
        ebit_df.loc[j] = ebit 
        ebit_after_tax = ebit * (1 - current_company["TaxRate"])
    
        sales_to_capital = []
        for i in range(10):
            sales_to_capital.append(drawn_stc)
        reinvestment_amt = []
        reinvestment_amt = np.array(reinvestment_amt)
        reinvestment_amt = revenue_growth_amt / sales_to_capital
        for i in range(10):
            reinvestment_df.loc[j] = reinvestment_amt
        
        #Calculate free cashflow to the firm
        ebit_after_tax = np.array(ebit_after_tax)
        FCFF = ebit_after_tax - reinvestment_amt
        fcff_df.loc[j] = FCFF
        #Calculate Cost of Capital
        Capital_Cost_start = drawn_wacc
        Capital_Cost_end = drawn_wacc_end
        Capital_Cost_subtract = (Capital_Cost_start-Capital_Cost_end)/5
        Capital_Cost = []
        for i in range(10):
            if i < 5:
                Capital_Cost.append(Capital_Cost_start)
            else:
                Capital_Cost.append(Capital_Cost_start-(i-4)*Capital_Cost_subtract)
    
        Discount_Factor = []
        for i in range(10):
            x = 1 / (Capital_Cost[i]+1)**(i+1)
            Discount_Factor.append(x)
            
        PV_FCFF = FCFF * Discount_Factor
        NPV_10 = sum(PV_FCFF)
    
    
        ###############################################################
    
        #Model TV Period
    
        ###############################################################
    
        growth_TV = drawn_revenue_end
        revenue_TV = revenue[-1] * (1 + drawn_revenue_end)
        ebit_TV = revenue_TV * op_margin[-1]
        ebit_after_tax_TV = ebit_TV * (1 - current_company["TaxRate"])
        reinvestment_TV = growth_TV / drawn_roic_end * ebit_after_tax_TV 
        FCFF_TV = ebit_after_tax_TV - reinvestment_TV
        CoC_TV = Capital_Cost[-1]
        
    
        TV = FCFF_TV / (CoC_TV - growth_TV)
        NPV_TV = TV * Discount_Factor[-1]
    
        ###############################################################
    
        #Calculating Firm Value, Equity Value and Share Price
    
        ###############################################################
    
        NPV_total = NPV_10 + NPV_TV
    
        Equity_Value = NPV_total - current_company["Debt"] + current_company["Cash"]
    
        PPS = Equity_Value / current_company["SharesOut"]
    
        value_list.append(PPS)
        
        invested_cap = []
        invested_cap.append(current_company["Invested Capital"])
        
        for i in range(10):
            x = invested_cap[i] + reinvestment_amt[i]
            invested_cap.append(x)
        
        avg_invested_cap = []
        
        for i in range(len(invested_cap)-1):
            x = (invested_cap[i] + invested_cap[i+1])/ 2
            avg_invested_cap.append(x)
        
        roic = ebit_after_tax / avg_invested_cap
        roic_df.loc[j] = roic


###############################################################

# Get Quantiles Dataframe

###############################################################

def get_quantiles_df():
    values_series = pd.Series(value_list)
    quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    quantile_values = values_series.quantile(quantiles)
    quantile_df = quantile_values.reset_index()
    quantile_df.columns = ['Quantile', 'Company Value']
    return quantile_df

###############################################################

# Build Site with Streamlit

###############################################################

st.header("Model Input")
#select company via streamlit 
select_company = st.radio("Select a company", list(companies.keys())) 
#set current company variable to selected
current_company = companies[select_company]

st.divider()
#Revenue Growth (lognormal)
st.subheader("Revenue Growth")
st.write("Estimate the initial revenue growth (constant for the first 5 years, then it converges towards the terminal value growth rate.\nHigher rates will expectedly lead to a higher company value.\nThe terminal value (stable) growth rate determines how fast cashflows grow in perpetuity.\nUsually this is set to a value that is at max the GDP growth rate. Company cannot indefinately grow faster than a country (Damodaran, 2025)")
st.write("")
st.write("")
st.write("")
st.write("We draw from a lognormal distribution, with mean and standard deviation")
mu_revenue_growth_percent = st.slider(
    "Your expected Revenue Growth",
    min_value=0.01,
    max_value=100.0,
    value=15.0,
    step=0.1,
    format="%.1f%%"
)

sigma_revenue_growth_percent = st.slider(
    "Your expected Standard Deviation",
    min_value=0.1,
    max_value=1/3 * mu_revenue_growth_percent, #we set for all STD the max to 1/3 mu because 3*STD captures 99.7% of the curve so 99.87% of draws will be positive, ensuring that we get resonable values for growth and wacc
    value=1/6 * mu_revenue_growth_percent,
    step=0.1,
    format="%.1f%%"
)
st.write("")
st.write("")
st.write("")
st.write("The terminal value of revenue growth is being drawn from a triangular distribution")
mu_revenue_end_percent = st.slider(
    "Your expected Revenue Growth End (upwards limited to initial revenue growth)",
    min_value=0.0,
    max_value=mu_revenue_growth_percent,
    value=1/2 * mu_revenue_growth_percent,
    step=0.1,
    format="%.1f%%")

st.divider()
st.subheader("WACC")
#get user input for revenue_growth via slider (lognormal)
st.write("The Capital Costs are the costs at which the cash flows are discounted, to get an equivalent to todays value. Initial cost of capital is calculated via the WACC formula. The terminal value is typically lower, because as companies mature, they tend to use more debt, which is cheaper than equity, bringing down the overall cost of capital.\n(In this model we assume the initial cost to be constant for 5 years, then it converges towards the terminal value cost of capital)")
st.write("")
st.write("")
st.write("")
st.write("We draw from a normal distribution, with mean and standard deviation")
mu_wacc_percent = st.slider(
    "Your expected WACC",
    min_value=0.1,
    max_value=100.0,
    value=15.0,
    step=0.1,
    format="%.1f%%"
)

#get user input for op margin via slider 
sigma_wacc_percent = st.slider(
    "Your expected Standard Deviation",
    min_value=0.1,
    max_value=1/3*mu_wacc_percent,
    value=1/6*mu_wacc_percent,
    step=0.1,
    format="%.1f%%"
)
st.write("")
st.write("")
st.write("")
st.write("The terminal value of WACC is being drawn from a triangular distribution")
mu_wacc_end_percent = st.slider(
    "Your expected terminal value (upwards limited by inital WACC value and lower limited by terminal revenue growth)",
    min_value=mu_revenue_end_percent,
    max_value=mu_wacc_percent,
    value=8.4,
    step=0.1,
    format="%.1f%%",
    help="If you get an error like left > mode or right < mode, check that the value you have selected here is between the values for the initial WACC and the TV growth"
)

st.divider()
#Operating Margin (Triangle)
st.subheader("Operating Margin")
st.write("The Target Operating Margin is a company's goal for operating profit as a percentage of revenue, reflecting its desired efficiency and profitability from core operations. A higher target margin typically indicates better scalability and cost control, which can significantly increase the company's valuation by boosting projected future earnings.")
st.write("We draw from a triangular distribution. The user is able to set a minimum and maximum for the distribution, but also a mode.")
#l_limit = st.number_input("Lower Limit Operating Margin:", min_value=0.0, max_value=100.0, value=10.0, step=0.1, format="%.2f")
#u_limit = st.number_input("Upper Limit Operating Margin:", min_value=l_limit, max_value=100.0, value=50.0, step=0.1, format="%.2f")

range_selection = st.slider(
    "Your expected range:",
    min_value=0.0,
    max_value=100.0,
    value=(30.0, 40.0),  # Startbereich (von, bis)
    step=0.01
)
l_limit = range_selection[0]
u_limit = range_selection[1]

mode_op_margin_percent = st.slider(
    "Your expected Operating Margin (used as mode of the distribution)",
    min_value=l_limit,
    max_value=u_limit,
    value=(l_limit + u_limit)/2,
    step=0.1,
    format="%.1f%%",
    help = " If you get an error like left > mode or right < mode, ensure that this value is in the range selected in the previous slider"
)
st.divider()
st.subheader("Sales to Capital")
st.write("The sales to capital ratio links growth and reinvestment. It shows how much you would earn using 1 USD in investment. We use it to calculate the total reinvestment required for the growth.")
st.caption("We draw from a triangular distribution based on company sectors (Software (System and Application): 1.71, Semiconductors: 1.10) Data per Damodaran). User is able to set the ratio.(https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/capex.html)", unsafe_allow_html=True)
s_to_c = st.slider(
    "Your expected Sales to Cap (used as mode of the distribution)",
    min_value=current_company["Industry STC"]-0.09,
    max_value=current_company["Industry STC"]+0.09,
    value=current_company["Industry STC"],
    step=0.01,
)
#https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/capex.html

st.divider()
st.subheader("ROIC")
st.write("Return of invested capital is the return of a company over its invested assets.To calculate the reinvestment, ROIC is required. The reinvestment rate in stable growth is equal to the stable growth rate of revenue over the return of invested capital. In this model it is used to link sales to capital reinvestment. (Damodaran, 2025)")
st.write("We draw from a normal distribution based on mean and sigma ")
mu_roic_percent = st.slider(
    "Your expected ROIC",
    min_value=0.1,
    max_value=100.0,
    value=20.0,
    step=0.1,
    format="%.1f%%"
)

sigma_roic_percent = st.slider(
    "Your expected Standard Deviation",
    min_value=0.1,
    max_value=1/2*mu_roic_percent,
    value=1/6*mu_roic_percent,
    step=0.1,
    format="%.1f%%"
)

st.divider()
st.subheader("Number of draws")
st.write("The amount of draws that is being done by the Monte-Carlo Simulation (efficiency-optimal approx. 1000)")
draws = st.number_input("Number of draws:", min_value=0, max_value=100000, value=1000, step=50)#, format="%.2f")
st.write("")
st.write("")
st.write("")
st.divider()
st.header("Model Output")
st.write("")

# set inputs to the decimal value
mu_wacc = mu_wacc_percent / 100
sigma_wacc = sigma_wacc_percent / 100
mu_wacc_end = mu_wacc_end_percent / 100
mu_revenue_growth = math.log(mu_revenue_growth_percent/100) # Log for lognormal
sigma_revenue_growth = (sigma_revenue_growth_percent / 100)
mu_revenue_end = mu_revenue_end_percent / 100
mode_op_margin = mode_op_margin_percent / 100
mu_roic = mu_roic_percent / 100
sigma_roic = sigma_roic_percent / 100


l_limit = l_limit / 100
u_limit = u_limit / 100

#call functions to rerun script on change
monte_carlo_EV()


plot_histo_two(sample_wacc, sample_wacc_t, title="WACC")
plot_histo(sample_opm, title="Operating Margin")

plot_histo_two(sample_rev, sample_rev_t, title ="Revenue Growth")

ticker = current_company["Ticker"]
try:
    stock_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
except Exception as e:
    st.markdown(f"### :green[Unable to connect to Yahoo Finance and show current stock price]")
    stock_price = 0
    
plot_histo(sorted(value_list)[:int(0.975*draws)], title="Share Value", price = stock_price)
st.caption(f"The current stock data is provided by Yahoo Finance\n(https://finance.yahoo.com/quote/{ticker})", unsafe_allow_html=True)
st.dataframe(get_quantiles_df())

st.write("")
st.subheader(f"{current_company["Ticker"]} Historical Charts")
plot_quant(pd.concat([pd.Series(current_company["Historic Revenue"]), revenue_df.mean()]),revenue_df, "Revenue")
ebit_margin_df = ebit_df / revenue_df
ebit_margin_comb = pd.Series(current_company["Historic EBIT"])/pd.Series(current_company["Historic Revenue"])
plot_quant(pd.concat([ebit_margin_comb, ebit_margin_df.mean()]),ebit_margin_df, "EBIT Margin")

plot_quant(pd.concat([pd.Series(current_company["Historic ROIC"])/100, roic_df.mean()]),roic_df, "ROIC")
st.caption(f"The historical stock data is provided by Bloomberg", unsafe_allow_html=False)