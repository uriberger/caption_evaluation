import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def collect_data():
    with open('metric_usage.csv', 'r') as fp:
        my_reader = csv.reader(fp)
        res = list(my_reader)
    res = res[1:]
    first_empty_line = min([i for i in range(len(res)) if sum([len(x) for x in res[i]]) == 0])
    res = res[:first_empty_line]
    return res
	
def analyze_data(data):
    metrics = list(set([x for outer in data for x in outer[3].split(', ') if x != '']))
    y2mcount = {}
    y2count = {}
    for sample in data:
        year = int(sample[0])
        cur_metrics = [x for x in sample[3].split(', ') if x != '']
        if year not in y2mcount:
            y2mcount[year] = {x: 0 for x in metrics}
        for metric in cur_metrics:
            y2mcount[year][metric] += 1
        if year not in y2count:
            y2count[year] = 0
        y2count[year] += 1
    metric_by_year = list(y2mcount.items())
    metric_by_year.sort(key=lambda x:x[0])
    m2ind = {}
    for metric in metrics:
        if metric == 'BLEU':
            m2ind[metric] = 0
        elif metric == 'METEOR':
            m2ind[metric] = 1
        elif metric == 'ROUGE':
            m2ind[metric] = 2
        elif metric == 'CIDEr':
            m2ind[metric] = 3
        elif metric == 'SPICe':
            m2ind[metric] = 4
        else:
            m2ind[metric] = 5
    m2pers = {x: [] for x in metrics}
    for year, metric_dict in metric_by_year:
        for metric, count in metric_dict.items():
            m2pers[metric].append(count/y2count[year])

    return metrics, y2mcount, y2count
	
def plot_metric_usage_per_year(metrics, y2mcount, window_size=None):
    similarity_metrics = ['BLEU', 'CIDEr', 'METEOR', 'ROUGE', 'SPICE', 'retrieval', 'CLIPScore', 'Named entities PR', 'Object PR', 'BERTScore', 'CHAIR', 'RefCLIPScore', 'WMD', 'exact match', 'SSD', 'TER', 'NW', 'NIST', 'PACScore', 'Noun overlap', 'fuzzy Noun overlap', 'RefPACScore', 'Semantic Score', 'Word PR', 'Rword', 'SPIPE', 'Verb PR', 'Noun PR', 'SemSim', 'fuzzy Verb overlap', 'SPICE-U', 'Clinical factual accuracy', 'Verb overlap', 'BLIP2Score', 'CLIPImageScore', 'MPNetScore', 'GPT4V evaluation']
    diversity_metrics = ['Div-n', 'mBLEU', 'novel', 'vocab', 'Self-CIDEr', 'unique', 'distinct', 'Self-BLEU', 'Tdiv', 'Dist', 'diversity edit dist', 'CIDErBtw', 'CLIP diversity', 'LSA']
    bias_metrics = ['Gender error', 'Gender ratio', 'BiasAmp', 'LIC']
    fluency_metrics = ['perplexity']
    syntactic_complexity_metrics = ['Yngve score']
    standalone_metrics = ['BLEU', 'CIDEr', 'METEOR', 'ROUGE', 'SPICE', 'retrieval', 'CLIPScore']
    window_size = None
    plt.clf()
    plt.figure().set_figheight(4)
    # Sort metrics by usage
    min_year = min(y2mcount.keys())
    max_year = max(y2mcount.keys())
    years = list(range(min_year, max_year+1))
    years = years[:-1]
    if window_size is not None:
        new_x_size = len(years) - window_size + 1
    metrics.sort(key=lambda x: sum([y2mcount[year][x] for year in years]), reverse=True)
    for metric in standalone_metrics:
        y = [y2mcount[year][metric] for year in years]
        if window_size is None:
            plt.plot(years, y, label=metric)
        else:
            smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
            smoothed_years = range(new_x_size)
            plt.plot(smoothed_years, smoothed_y, label=metric)
    other_similarity_metrics = [x for x in similarity_metrics if x not in standalone_metrics]
    other_similarity_str = f'Other similarity metrics (N={len(other_similarity_metrics)})'
    other_similarity_y = [sum([y2mcount[year][x] for x in other_similarity_metrics]) for year in years]
    fluency_str = f'Fluency metrics (N={len(fluency_metrics)})'
    fluency_y = [sum([y2mcount[year][x] for x in fluency_metrics]) for year in years]
    diversity_str = f'Diversity metrics (N={len(diversity_metrics)})'
    diversity_y = [sum([y2mcount[year][x] for x in diversity_metrics]) for year in years]
    bias_str = f'Bias metrics (N={len(bias_metrics)})'
    bias_y = [sum([y2mcount[year][x] for x in bias_metrics]) for year in years]
    if window_size is None:
        plt.plot(years, other_similarity_y, label=other_similarity_str)
        plt.plot(years, fluency_y, label=fluency_str, linestyle='dashed')
        plt.plot(years, diversity_y, label=diversity_str, linestyle='dotted')
        plt.plot(years, bias_y, label=bias_str, linestyle='dashdot', color='black')
    else:
        smoothed_years = range(new_x_size)
        smoothed_y = np.convolve(other_similarity_y, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_years, smoothed_y, label=other_similarity_str)
        smoothed_y = np.convolve(fluency_y, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_years, smoothed_y, label=fluency_str, linestyle='dashed')
        smoothed_y = np.convolve(diversity_y, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_years, smoothed_y, label=diversity_str, linestyle='dotted')
        smoothed_y = np.convolve(bias_y, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smoothed_years, smoothed_y, label=bias_str, linestyle='dashdot', color='black')
    plt.legend(ncol=3, loc='center left', bbox_to_anchor=(-0.05, -0.25))
    plt.xlabel('Year')
    plt.ylabel('# of papers that used the metric')
    if window_size is not None:
        plt.xticks(ticks=[2*x*(new_x_size/len(years)) for x in range(math.ceil(len(years)/2))], labels=[years[2*x] for x in range(math.ceil(len(years)/2))])
    plt.savefig('metric_usage_per_year.png', bbox_inches='tight')
	
def plot_papers_per_community(data, y2mcount, window_size=3):
    conf2community = {'ECCV': 'Vision', '*SEM': 'NLP', 'TACL': 'NLP', 'ICML': 'Machine Learning', 'EACL': 'NLP', 'CVPR': 'Vision', 'EMNLP': 'NLP', 'Neurips': 'Machine Learning', 'AAAI': 'Machine Learning', 'ACL': 'NLP', 'ICCV': 'Vision', 'AACL': 'NLP', 'NAACL': 'NLP', 'CONLL': 'NLP', 'ICLR': 'Machine Learning'}
    communities = list(set(conf2community.values()))
    community2year_count = {x: {i: 0 for i in range(2010, 2024)} for x in communities}
    community2color = {'Machine Learning': 'blue', 'NLP': 'orange', 'Vision': 'green'}
    plt.clf()
    min_year = min(y2mcount.keys())
    max_year = max(y2mcount.keys())
    years = list(range(min_year, max_year+1))
    years.remove(2024)
    for sample in data:
        year = int(sample[0])
        if year == 2024:
            continue
        conf = sample[4]
        conf_name = conf.split()[0]
        assert conf_name in conf2community, f'Found no community for conference {conf_name}'
        community2year_count[conf2community[conf_name]][year] += 1
    new_x_size = len(years) - window_size + 1
    for community in communities:
        y = [community2year_count[community][year] for year in years]
        if window_size is None:
            plt.plot(years, y, label=community)
        else:
            smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
            smoothed_years = range(new_x_size)
            plt.plot(smoothed_years, smoothed_y, label=community, color=community2color[community])
    #plt.axvline(2014, label='MSCOCO publication', color='red')
    plt.legend()
    if window_size is not None:
        plt.xticks(ticks=[2*x*(new_x_size/len(years)) for x in range(math.ceil(len(years)/2))], labels=[years[2*x] for x in range(math.ceil(len(years)/2))])
    plt.xlabel('Year')
    plt.ylabel('# of papers')
    plt.savefig('papers_per_community.png')
	
def plot_mean_metric_num_per_year(y2mcount, y2count, window_size=None):
	plt.clf()
	min_year = min(y2mcount.keys())
	max_year = max(y2mcount.keys())
	years = list(range(min_year, max_year+1))
	years = years[:-1]
	if window_size is not None:
		new_x_size = len(years) - window_size + 1
	y2mean_metric_num = {year: sum(y2mcount[year].values())/y2count[year] for year in years}
	#errs = [statistics.stdev(y2mcount[year].values()) for year in years]
	#plt.errorbar(years, [y2mean_metric_num[year] for year in years], yerr=errs, capsize=5, capthick=1)
	y = [y2mean_metric_num[year] for year in years]
	if window_size is None:
		plt.plot(years, y, color='black')
	else:
		smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
		smoothed_years = range(new_x_size)
		plt.plot(smoothed_years, smoothed_y, color='black')
	if window_size is not None:
		plt.xticks(ticks=[2*x*(new_x_size/len(years)) for x in range(math.ceil(len(years)/2))], labels=[years[2*x] for x in range(math.ceil(len(years)/2))])
	plt.xlabel('Year')
	plt.ylabel('Mean number of metric used')
	plt.ylim((-0.5, 5.5))
	plt.savefig('mean_metric_num_per_year.png')
	
def plot_human_evaluation_per_year(data, y2count, window_size=None):
    plt.clf()
    min_year = min(y2count.keys())
    max_year = max(y2count.keys())
    years = list(range(min_year, max_year+1))
    if window_size is not None:
        new_x_size = len(years) - window_size + 1
    y2he_count = {x: 0 for x in years}
    for x in data:
        if len(x[5].strip()) > 0:
            y2he_count[int(x[0])] += 1
    y = [y2he_count[year]/y2count[year] for year in years]
    if window_size is None:
        plt.plot(years, y, color='black')
    else:
        smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        smoothed_years = range(new_x_size)
        plt.plot(smoothed_years, smoothed_y, color='black')
    if window_size is not None:
        plt.xticks(ticks=[2*x*(new_x_size/len(years)) for x in range(math.ceil(len(years)/2))], labels=[years[2*x] for x in range(math.ceil(len(years)/2))])
    plt.xlabel('Year')
    plt.ylabel('Fraction of paper that used human evaluation')
    plt.savefig('human_evaluation_per_year.png')
	
def plot_framework_usage_per_year(data, y2mcount, window_size=4):
    frameworks = list(set([x for outer in data for x in outer[5].split(', ') if len(outer[5].strip()) > 0]))
    min_year = min(y2mcount.keys())
    max_year = max(y2mcount.keys())
    years = list(range(min_year, max_year+1))
    years = years[:-1]
    new_x_size = len(years) - window_size + 1
    framework2year_count = {x: {year: 0 for year in years} for x in frameworks}
    for sample in data:
        if len(sample[5].strip()) == 0:
            continue
        year = int(sample[0])
        if year == 2024:
            continue
        cur_frameworks = sample[5].split(', ')
        for framework in cur_frameworks:
            framework2year_count[framework][year] += 1
    plt.clf()
    # Sort frameworks by usage
    frameworks.sort(key=lambda x: sum([framework2year_count[x][year] for year in years]), reverse=True)
    for framework in frameworks:
        y = [framework2year_count[framework][year] for year in years]
        if window_size is None:
            plt.plot(years, y, label=framework)
        else:
            smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
            smoothed_years = range(new_x_size)
            plt.plot(smoothed_years, smoothed_y, label=framework)
    if window_size is not None:
        plt.xticks(ticks=[2*x*(new_x_size/len(years)) for x in range(math.ceil(len(years)/2))], labels=[years[2*x] for x in range(math.ceil(len(years)/2))])
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('# of papers that used the framework')
    plt.savefig('framework_usage_per_year.png')

if __name__ == '__main__':
    data = collect_data()
    metrics, y2mcount, y2count = analyze_data(data)
    plot_metric_usage_per_year(metrics, y2mcount)
    plot_papers_per_community(data, y2mcount)
    plot_mean_metric_num_per_year(y2mcount, y2count)
    plot_human_evaluation_per_year(data, y2count)
    plot_framework_usage_per_year(data, y2mcount)
