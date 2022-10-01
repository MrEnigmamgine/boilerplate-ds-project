import pandas as pd

def anova_variance_in_target_for_cat(df, target, cat, alpha=0.05):
    """Quickly test a target against the categories in another column."""
    from scipy.stats import f_oneway
    s= df[cat]
    vals = s.sort_values().unique()
    subsets = [df[s == vals[x]][target] for x, v in enumerate(vals)]
    stat, p = f_oneway(*subsets)
    result={'reject': p < alpha,
            'h0' : f"There is no variance in {target} between subsets of {cat}",
            'stat_name': 'F',
            'stat': stat,
            'p_value': p,
            'alpha': alpha
        }
    return result

def ttest_target_for_each_cat(df, target, cat, alpha=0.05):
    """Quickly test each category subset against the overall mean to identify which categories are signficant."""
    from scipy.stats import ttest_1samp
    s= df[cat]
    vals = s.sort_values().unique()
    subsets = [df[s == vals[x]][target] for x, v in enumerate(vals)]
    mean = df[target].mean()
    out = {}
    for i, subset in enumerate(subsets):
        stat, p = ttest_1samp(subset, mean)
        result={'reject': p < alpha,
                'h0' : f"The mean of {target} for {cat}:{vals[i]} is the same as the overall population",
                'stat_name': 'F',
                'stat': stat,
                'p_value': p,
                'alpha': alpha
            }
        out[str(vals[i])] = result
    return out

def chi2_test(s1, s2, alpha=0.05):
    """Quickly determine if two samples are dependent of one another."""
    from scipy.stats import chi2_contingency
    table = pd.crosstab(s1, s2)
    stat, p, dof, expected = chi2_contingency(table)
    result={
        'reject': p < alpha,
        'h0' : f"The two samples are independent.",
        'stat_name': 'Chi2',
        'stat': stat,
        'p_value': p,
        'alpha': alpha,
        # 'misc' : {
            # 'dof' : dof,
            # 'expected': expected,
            # 'observed': table
            # }
        }
    return result

def spearman_correllation_test(df, x, y, alpha=0.05):
    from scipy.stats import spearmanr

    stat, p = spearmanr(df[x], df[y])
    result={'reject': p < alpha,
        'h0' : f"The samples of '{x}' and '{y}' are independant",
        'stat_name': 'correlation',
        'stat': stat,
        'p_value': p,
        'alpha': alpha
    }
    return result

def pearson_correllation_test(df, x, y, alpha=0.05):
    from scipy.stats import pearsonr

    stat, p = pearsonr(df[x], df[y])
    result={'reject': p < alpha,
        'h0' : f"The samples of '{x}' and '{y}' are independant",
        'stat_name': 'correlation',
        'stat': stat,
        'p_value': p,
        'alpha': alpha
    }
    return result

def shapiro_gausian_test(s, alpha=0.05):
    from scipy.stats import shapiro

    stat, p = stat, p = shapiro(s)
    result={'reject': p < alpha,
        'h0' : f"The distribution is gaussian",
        'stat_name': 'statistic',
        'stat': stat,
        'p_value': p,
        'alpha': alpha
    }
    return result

def all_the_stats(df, override_categorical=[], override_numerical=[]):
    # Initialize the dictionary that will be iteratively built
    out = {}
    # Separate columns into categorical and numerical
    coltype = get_column_types(df, override_categorical=override_categorical)
    
    # Loop through each categorical column
    for col in coltype['cat']:
        if len(df[col].value_counts()) > 1:
            cold = out[col] = {}
            
            # Run a chi2 test on every other categorical column
            this = cold['chi2'] = {}
            for target in coltype['cat']:
                if len(df[target].value_counts()) > 1:
                    if target != col:
                        this[target] = chi2_test(df[col], df[target])
            # Run an anova test on every numerical column
            this = cold['anova'] = {}
            for target in coltype['num']:
                if target != col:
                    anova = this[target] = anova_variance_in_target_for_cat(df, target, col)
                    # If we reject the null run a ttest to determine which categories are significant
                    if anova['reject'] == True:
                        anova['ttest'] = ttest_target_for_each_cat(df, target, col)

    # The loop through each numerical column
    for col in coltype['num']:
        
        cold = out[col] = {}
        
        # Repeat the Anova tests on each categorical column for readability
        this = cold['anova'] = {}
        for target in coltype['cat']:
            if len(df[target].value_counts()) > 1:
                if target != col:
                    anova = this[target] = anova_variance_in_target_for_cat(df, col, target)
                    if anova['reject'] == True:
                        anova['ttest'] = ttest_target_for_each_cat(df, col, target)

        # Run a correlation test for every other numerical column
        this = cold['spearmanr'] = {}
        for target in coltype['num']:
            if target != col:
                this[target] = spearman_correllation_test(df, target, col)

    return out