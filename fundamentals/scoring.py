class Fundamentals:
    # --- Loading fundamental Data ---
    def compute_fundamental_score(self,info):
        """
        Score a stock 0-100 for investment quality using forward-looking ratios.
        Returns (score, breakdown_dict).
        """
        score = 0
        breakdown = {}

        # 1. Forward P/E (lower is better, <15 ideal, <25 acceptable)
        fpe = info.get('forwardPE')
        if fpe and fpe > 0:
            if fpe < 15:   pts = 20
            elif fpe < 20: pts = 15
            elif fpe < 25: pts = 10
            elif fpe < 35: pts = 5
            else:           pts = 0
            score += pts
            breakdown['Forward P/E'] = (round(fpe, 1), pts, 20)
        else:
            breakdown['Forward P/E'] = ('N/A', 0, 20)

        # 2. PEG Ratio (price/earnings-to-growth; <1 = undervalued)
        peg = info.get('pegRatio')
        if peg and peg > 0:
            if peg < 1:    pts = 20
            elif peg < 1.5: pts = 15
            elif peg < 2:  pts = 8
            else:           pts = 0
            score += pts
            breakdown['PEG Ratio'] = (round(peg, 2), pts, 20)
        else:
            breakdown['PEG Ratio'] = ('N/A', 0, 20)

        # 3. Return on Equity (higher is better; >15% good)
        roe = info.get('returnOnEquity')
        if roe is not None:
            roe_pct = roe * 100
            if roe_pct > 20:   pts = 20
            elif roe_pct > 15: pts = 15
            elif roe_pct > 10: pts = 8
            elif roe_pct > 0:  pts = 3
            else:               pts = 0
            score += pts
            breakdown['Return on Equity'] = (f"{roe_pct:.1f}%", pts, 20)
        else:
            breakdown['Return on Equity'] = ('N/A', 0, 20)

        # 4. Debt-to-Equity (lower is better; <0.5 ideal)
        de = info.get('debtToEquity')
        if de is not None:
            if de < 30:    pts = 20
            elif de < 60:  pts = 15
            elif de < 100: pts = 8
            elif de < 150: pts = 3
            else:           pts = 0
            score += pts
            breakdown['Debt / Equity'] = (f"{de:.0f}%", pts, 20)
        else:
            breakdown['Debt / Equity'] = ('N/A', 0, 20)

        # 5. Revenue Growth (YoY; >10% strong)
        rev_growth = info.get('revenueGrowth')
        if rev_growth is not None:
            rg = rev_growth * 100
            if rg > 20:   pts = 20
            elif rg > 10: pts = 15
            elif rg > 5:  pts = 8
            elif rg > 0:  pts = 3
            else:          pts = 0
            score += pts
            breakdown['Revenue Growth (YoY)'] = (f"{rg:.1f}%", pts, 20)
        else:
            breakdown['Revenue Growth (YoY)'] = ('N/A', 0, 20)

        return score, breakdown
    
    def score_to_rating(self,score):
        if score >= 80: return "⭐⭐⭐⭐⭐ Strong Buy"
        if score >= 65: return "⭐⭐⭐⭐ Buy"
        if score >= 50: return "⭐⭐⭐ Hold"
        if score >= 35: return "⭐⭐ Weak"
        return "⭐ Avoid"

    def score_to_color(self,score):
        if score >= 65: return "#00e676"
        if score >= 50: return "#ffb74d"
        return "#ff1744"
    