# 📊 Project Finance Indicators

A Python tool for calculating and visualizing project financial indicators.

## 📝 Description

This code is completely free and can be modified without any restriction.

This project provides a comprehensive suite of tools for calculating and visualizing key financial indicators used in project evaluation. The concepts are presented and explained in the article "QUELQUES NOTIONS DE BASE DE LA FINANCE DES PROJETS" available here:
- [basic_finance_elements-v3.pdf](https://www.anigraphics.fr/introduction/math_finance/basic_finance_elements-v3.pdf)
- [LinkedIn Article](https://www.linkedin.com/feed/update/urn:li:activity:6576694086428827648/)

To have a good understanding of these financial terms technically, it is recommended to read the article mentioned above.

## 🔍 Features

### Financial Indicators
- **PV (Present Value)**: Calculate the present value of future cash flows
- **FV (Future Value)**: Calculate the future value of current investments
- **NPV (Net Present Value)**: Evaluate project profitability
- **IRR (Internal Rate of Return)**: Calculate the project's internal rate of return
- **TEG (Global Effective Rate)**: Determine the global effective interest rate
- **MIRR (Modified Internal Rate of Return)**: Calculate the modified internal rate of return
- **PBP (Payback Period)**: Determine how long it takes to recover the investment
- **DPBP (Discounted Payback Period)**: Calculate the discounted payback period
- **PI (Profitability Index)**: Measure relative profitability of investments

### Advanced Features
- **Amortization Tables**: Generate detailed amortization schedules
- **Visualizations**: Automatic plotting of all key indicators
- **Automatic Export**:
  - Graphs are saved in `./output/graphs`
  - Data is exported as CSV in `./output/data`

## 🛠️ Installation

1. **Create a virtual environment**:
```bash
python -m venv venv
```

2. **Activate the virtual environment**:
- **Windows**:
```bash
venv\Scripts\activate.bat
```
- **Unix/MacOS**:
```bash
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install numpy pandas scipy sympy matplotlib numpy_financial openpyxl
```

## 📂 Project Structure

```
project-finance-indicators/
├── LICENSE
├── MANIFEST.in
├── README.md
├── pyproject.toml
├── requirements.txt
├── setup.py
├── src/
│   ├── main.py
│   └── modules/
│       ├── __init__.py
│       ├── data_manager.py
│       ├── financial_calculator.py
│       ├── financial_examples.py
│       ├── graph_plotter.py
│       └── loan_calculator.py
└── output/
    ├── data/
    │   └── (Generated CSV files)
    └── plots/
        └── (Generated plot images)
```

## 📚 Usage

### Basic Execution
Run from src directory this command-line:
```bash
python main.py
```

### Command Line Arguments for table amortization generation
Run from src directory this command-line:
```bash
python main.py --rate <loan rate in %> --n_periods <number of periods> --principal <loan principal> --when <'begin' or 'end'>
```


**Arguments:**
- **rate**: Annual interest rate as % value (ex: 6 for 6%)
- **n_periods**: Number of loan periods (in months or years)
- **principal**: Loan principal
- **when**: When payments are made (begin or end)

**Example:**
```bash
python main.py --rate 6 --n_periods 48 --principal 100000 --when 'end'
```

### command line arguments for examples execution
```bash
python main.py --examples
```

### Available Examples
The `main.py` file contains several practical examples:
- `example_1_1_1_1`: Solving complex financial equations
- `example_1_1_1`: Calculating IRR, MIRR, and payback periods
- `example_5_1`: Project comparison using NPV
- `example_7_2`: Generating amortization tables

### Output Structure
- **Graphs**: Saved in `./output/graphs` with timestamps
- **Data**: Exported as CSV in `./output/data` with timestamps
- **Console**: Detailed display of calculations and results

## 📊 Visualization Examples
- Cumulative cash flow evolution
- NPV/IRR project comparisons
- Amortization tables with payment breakdowns
- Break-even points and recovery periods

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## ✍️ Author
**Abdel YEZZA, Ph.D**

## 📄 License
This code is completely free and can be modified without any restriction.
