def get_color(display):
    if display['strategy'] == 'mv':
        if display['sample'] == 'train' and display['robust_pred'] is True:
            return 'darkorange'
        else:
            return 'red'
    elif display['strategy'] == 'dro':
        if display['sample'] == 'train' and display['robust_pred'] is True:
            return 'lightblue'
        else:
            return 'blue'
    elif display['strategy'] == 'dro_l':
        if display['sample'] == 'train' and display['robust_pred'] is True:
            return 'lightgreen'
        else:
            return 'green'


def get_color_rets(d, display):
    nb = {'mv': 0, 'dro': 0, 'dro_l': 0}
    position = 0
    for dd in display:
        nb[dd['strategy']] += 1
        if not dd == d:
            position += 1
        else:
            print('EQUALITY BETWEEN Ds WORKS')


def get_label(display, plot='bar'):
    """
    plot: 'bar', 'reli', 'risk'
    """
    rad = f'_{display["radius"]}'
    if display['robust_pred']:
        rob = '_robpred'
    else:
        rob = ''
        if display['strategy'] == 'mv':
            rad = ''

    if plot in ['reli', 'risk']:
        return display['strategy'] + '_' + rad
    if plot == 'bar':
        return display['strategy'] + '_' + display['sample'] + rad + rob


display_2_bar = [{'strategy': 'mv', 'sample': 'train', 'radius': 0.0005, 'robust_pred': False},
                 {'strategy': 'mv', 'sample': 'train', 'radius': 0.0005, 'robust_pred': True},
                 {'strategy': 'mv', 'sample': 'test', 'radius': 0.0005, 'robust_pred': False},

                 {'strategy': 'dro_l', 'sample': 'train', 'radius': 0.0005, 'robust_pred': False},
                 {'strategy': 'dro_l', 'sample': 'train', 'radius': 0.0005, 'robust_pred': True},
                 {'strategy': 'dro_l', 'sample': 'test', 'radius': 0.0005, 'robust_pred': False},

                 {'strategy': 'dro', 'sample': 'test', 'radius': 0.0005, 'robust_pred': False},
                 {'strategy': 'dro', 'sample': 'train', 'radius': 0.0005, 'robust_pred': False},
                 {'strategy': 'dro', 'sample': 'train', 'radius': 0.0005, 'robust_pred': True}
                 ]

exp_14 = {'years_30': False, 'radii': [0.0005, 0.000001], 'output_name': 'exp_14',
          'method': 'bootstrap', 'sample_size': 5000, 'test_years': 2, 'train_years': 10}

display_15_bar = [{'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                  {'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},
                  {'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 1e-6, 'robust_pred': True},

                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},
                  {'strategy': 'dro_l', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 1e-5, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 1e-5, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 1e-5, 'robust_pred': True}
                  ]

display_15_rets = [{'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                   {'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                   {'strategy': 'dro', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},
                   {'strategy': 'dro', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},

                   {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                   {'strategy': 'dro_l', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                   {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},
                   {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False}
                   ]

display_15_reli = [{'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                   {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                   {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},
                   {'strategy': 'dro', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False}]

exp_16 = {'years_30': False,
          'radii': [5e-5, 1e-5, 5e-6, 1e-6, 5e-7],
          'output_name': 'exp_16',
          'method': 'bootstrap',
          'sample_size': 5000,
          'test_years': 2,
          'train_years': 10,
          'target_yearly': 0.1,
          'obj': 'MaxRet',
          'rm': 'MV'}

exp_17 = {'years_30': False,
          'radii': [1e-5, 5e-6, 1e-6],
          'output_name': 'exp_17',
          'method': 'bootstrap',
          'sample_size': 5000,
          'test_years': 2,
          'train_years': 10,
          'target_yearly': 0.075,
          'obj': 'MaxRet',
          'risk_measure': 'std'}  # CVaR

exp_1 = {'years_30': False,
         'radii': [1e-5, 5e-6, 1e-6],
         'output_name': 'exp_test',
         'method': 'bootstrap',
         'sample_size': 5000,
         'test_years': 2,
         'train_years': 10,
         'target_yearly': 0.075,
         'obj': 'MaxRet',
         'risk_measure': 'std'}  # CVaR

exp_2 = {'years_30': False,
         'radii': [1e-5, 5e-6, 1e-6],
         'output_name': 'exp_2',
         'method': 'bootstrap',
         'sample_size': 15000,
         'test_years': 2,
         'train_years': 10,
         'target_yearly': 0.075,
         'obj': 'MaxRet',
         'risk_measure': 'std'}  # CVaR

# ----------------------------------------------    max return   --------------------------------------------

exp_max_ret = {'years_30': False,
               'radii': [5e-5, 1e-5, 5e-6, 1e-6, 1e-7, 1e-8, 1e-9],
               'output_name': 'exp_max_ret',
               'method': 'bootstrap',
               'sample_size': 15000,
               'test_years': 2,
               'train_years': 10,
               'target_yearly': 0.075,
               'obj': 'MaxRet',
               'risk_measure': 'std'}

rets_max_ret = [{'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                {'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                {'strategy': 'dro', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 1e-7, 'robust_pred': False},

                # {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                # {'strategy': 'dro_l', 'sample': 'test',  'radius': 5e-6, 'robust_pred': False},

                {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},

                {'strategy': 'dro', 'sample': 'test', 'radius': 5e-5, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 5e-5, 'robust_pred': False}]

rets_l_max_ret = [{'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                {'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-7, 'robust_pred': False},

                {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'test',  'radius': 5e-6, 'robust_pred': False},

                {'strategy': 'dro_l', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},

                {'strategy': 'dro_l', 'sample': 'test', 'radius': 5e-5, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-5, 'robust_pred': False}]

bar_max_ret = [{'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
               {'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},
               {'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

               {'strategy': 'dro', 'sample': 'test', 'radius': 1e-9, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-9, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-9, 'robust_pred': True},

               {'strategy': 'dro', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-7, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-7, 'robust_pred': True},

               {'strategy': 'dro', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-6, 'robust_pred': True},

               {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},

               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-5, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-5, 'robust_pred': True},
               {'strategy': 'dro', 'sample': 'test', 'radius': 1e-5, 'robust_pred': False},

               {'strategy': 'dro', 'sample': 'test', 'radius': 5e-5, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 5e-5, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 5e-5, 'robust_pred': True}]

bar_l_max_ret = [{'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
               {'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},
               {'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

               {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-9, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-9, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-9, 'robust_pred': True},

               {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-7, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-7, 'robust_pred': True},

               {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-6, 'robust_pred': True},

               {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},
               {'strategy': 'dro_l', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

               {'strategy': 'dro_l', 'sample': 'test', 'radius': 5e-5, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-5, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-5, 'robust_pred': True}]

violin_max_ret = [{'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 5e-5, 'robust_pred': False}]

# ----------------------------------------------    min risk   --------------------------------------------

exp_min_risk = {'years_30': False,
                'radii': [1e-4, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 1e-8, 1e-9],
                'output_name': 'exp_min_risk',
                'method': 'bootstrap',
                'sample_size': 2000,
                'test_years': 2,
                'train_years': 10,
                'target_yearly': 0.075,
                'obj': 'MinRisk',
                'risk_measure': 'std'}  # CVaR

exp_min_risk_10 = {'years_30': False,
                'radii': [1e-4, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7, 1e-8, 1e-9],
                'output_name': 'exp_min_risk_10',
                'method': 'bootstrap',
                'sample_size': 2000,
                'test_years': 2,
                'train_years': 10,
                'target_yearly': 0.10,
                'obj': 'MinRisk',
                'risk_measure': 'std'}

rets_min_risk = [{'strategy': 'mv', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
                 {'strategy': 'mv', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},

                 {'strategy': 'dro', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},
                 {'strategy': 'dro', 'sample': 'train', 'radius': 1e-7, 'robust_pred': False},

                 {'strategy': 'dro', 'sample': 'test', 'radius': 5e-7, 'robust_pred': False},
                 {'strategy': 'dro', 'sample': 'train', 'radius': 5e-7, 'robust_pred': False},

                 {'strategy': 'dro', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},
                 {'strategy': 'dro', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False}]

rets_l_min_risk = [{'strategy': 'mv', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
                {'strategy': 'mv', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},

                {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-7, 'robust_pred': False},

                {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-7, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'test',  'radius': 5e-7, 'robust_pred': False},

                {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False}]

bar_min_risk = [{'strategy': 'mv', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
               {'strategy': 'mv', 'sample': 'train', 'radius': 1e-6, 'robust_pred': True},
               {'strategy': 'mv', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},

               {'strategy': 'dro', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-7, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-7, 'robust_pred': True},

                {'strategy': 'dro', 'sample': 'train', 'radius': 5e-7, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 5e-7, 'robust_pred': True},
                {'strategy': 'dro', 'sample': 'test', 'radius': 5e-7, 'robust_pred': False},

               {'strategy': 'dro', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-6, 'robust_pred': True},

                {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},

                {'strategy': 'dro', 'sample': 'test', 'radius': 1e-5, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 1e-5, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 1e-5, 'robust_pred': True}]

bar_l_min_risk = [{'strategy': 'mv', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
               {'strategy': 'mv', 'sample': 'train', 'radius': 1e-6, 'robust_pred': True},
               {'strategy': 'mv', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},

                  {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},
                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-7, 'robust_pred': False},
                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-7, 'robust_pred': True},

                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-7, 'robust_pred': False},
                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-7, 'robust_pred': True},
                  {'strategy': 'dro_l', 'sample': 'test', 'radius': 5e-7, 'robust_pred': False},

                  {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},
                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-6, 'robust_pred': True},

                  {'strategy': 'dro_l', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},
                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},

                  {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-5, 'robust_pred': False},
                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-5, 'robust_pred': False},
                  {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-5, 'robust_pred': True}]

violin_min_risk = [{'strategy': 'mv', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 1e-5, 'robust_pred': False}]

violin_l_min_risk = [{'strategy': 'mv', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},

                  {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},

                  {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},

                  {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-5, 'robust_pred': False}]

# -----------------------------------------------  cvar tg  -----------------------------------------------------

exp_cvar_tg = {'years_30': False,
               'radii': [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 5e-6, 1e-6, 1e-7, 1e-8, 1e-9],
               'output_name': 'exp_cvar_tg',
               'method': 'bootstrap',
               'sample_size': 15000,
               'test_years': 2,
               'train_years': 10,
               'target_yearly': 0.05,
               'obj': 'MinRisk',
               'risk_measure': 'CVaR'}

bar_cvar_tg = [{'strategy': 'mv', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
               {'strategy': 'mv', 'sample': 'train', 'radius': 1e-6, 'robust_pred': True},
               {'strategy': 'mv', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},

               {'strategy': 'dro', 'sample': 'test', 'radius': 1e-8, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-8, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-8, 'robust_pred': True},

                {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},
                {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

               {'strategy': 'dro', 'sample': 'test', 'radius': 1e-5, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-5, 'robust_pred': False},
               {'strategy': 'dro', 'sample': 'train', 'radius': 1e-5, 'robust_pred': True},

                {'strategy': 'dro', 'sample': 'test', 'radius': 1e-4, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 1e-4, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 1e-4, 'robust_pred': True},

                {'strategy': 'dro', 'sample': 'test', 'radius': 1e-1, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 1e-1, 'robust_pred': False},
                {'strategy': 'dro', 'sample': 'train', 'radius': 1e-1, 'robust_pred': True}]

bar_l_cvar_tg = [{'strategy': 'mv', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
               {'strategy': 'mv', 'sample': 'train', 'radius': 1e-6, 'robust_pred': True},
               {'strategy': 'mv', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},

               {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-8, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-8, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-8, 'robust_pred': True},

                {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},
                {'strategy': 'dro_l', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

               {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-5, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-5, 'robust_pred': False},
               {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-5, 'robust_pred': True},

                {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-4, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-4, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-4, 'robust_pred': True},

                {'strategy': 'dro_l', 'sample': 'test', 'radius': 1e-1, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-1, 'robust_pred': False},
                {'strategy': 'dro_l', 'sample': 'train', 'radius': 1e-1, 'robust_pred': True}]

display_violin_cvar_tg = [{'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                         {'strategy': 'dro', 'sample': 'test', 'radius': 1e-8, 'robust_pred': False},

                         {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                         {'strategy': 'dro', 'sample': 'test', 'radius': 1e-1, 'robust_pred': False}]

# -----------------------------------------------  many cvars  -----------------------------------------------------------


exp_TEST_cvar = {'years_30': False,
                 'radii': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6],
                 'output_name': 'exp_TEST_cvar',
                 'method': 'bootstrap',
                 'sample_size': 5000,
                 'test_years': 2,
                 'train_years': 10,
                 'target_yearly': None,
                 'obj': 'MinRisk',
                 'risk_measure': 'CVaR'}  # CVaR

bar_cvar = [{'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                  {'strategy': 'mv', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},
                  {'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 1e-6, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 1e-6, 'robust_pred': True},

                  {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 5e-6, 'robust_pred': True},
                  {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 5e-5, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 5e-5, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 5e-5, 'robust_pred': True},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 5e-4, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 5e-4, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 5e-4, 'robust_pred': True}]

exp_TEST_cvar_2 = {'years_30': False,
                   'radii': [5e-7, 1e-7, 1e-9],
                   'output_name': 'exp_TEST_cvar_2',
                   'method': 'bootstrap',
                   'sample_size': 5000,
                   'test_years': 2,
                   'train_years': 10,
                   'target_yearly': None,
                   'obj': 'MinRisk',
                   'risk_measure': 'CVaR'}  # CVaR

bar_cvar_2 = [{'strategy': 'mv', 'sample': 'train', 'radius': 1e-9, 'robust_pred': False},
                  {'strategy': 'mv', 'sample': 'train', 'radius': 1e-9, 'robust_pred': True},
                  {'strategy': 'mv', 'sample': 'test', 'radius': 1e-9, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 1e-9, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 1e-9, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 1e-9, 'robust_pred': True},

                  {'strategy': 'dro', 'sample': 'train', 'radius': 1e-7, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 1e-7, 'robust_pred': True},
                  {'strategy': 'dro', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 5e-7, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 5e-7, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 5e-7, 'robust_pred': True}]

exp_TEST_cvar_3 = {'years_30': False,
                   'radii': [10, 1, 0.1],
                   'output_name': 'exp_TEST_cvar_3',
                   'method': 'bootstrap',
                   'sample_size': 5000,
                   'test_years': 2,
                   'train_years': 10,
                   'target_yearly': None,
                   'obj': 'MinRisk',
                   'risk_measure': 'CVaR'}  # CVaR

bar_cvar_3 = [{'strategy': 'mv', 'sample': 'train', 'radius': 0.1, 'robust_pred': False},
                  {'strategy': 'mv', 'sample': 'train', 'radius': 0.1, 'robust_pred': True},
                  {'strategy': 'mv', 'sample': 'test', 'radius': 0.1, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 0.1, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 0.1, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 0.1, 'robust_pred': True},

                  {'strategy': 'dro', 'sample': 'train', 'radius': 1, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 1, 'robust_pred': True},
                  {'strategy': 'dro', 'sample': 'test', 'radius': 1, 'robust_pred': False},

                  {'strategy': 'dro', 'sample': 'test', 'radius': 10, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 10, 'robust_pred': False},
                  {'strategy': 'dro', 'sample': 'train', 'radius': 10, 'robust_pred': True}]

display_violin = [{'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                         {'strategy': 'dro', 'sample': 'test', 'radius': 1e-5, 'robust_pred': False},

                         {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                         {'strategy': 'dro', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False}]

display_violin_TEST_2 = [{'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                         {'strategy': 'dro', 'sample': 'test', 'radius': 1e-5, 'robust_pred': False},

                         {'strategy': 'dro', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                         {'strategy': 'dro', 'sample': 'test', 'radius': 1e-6, 'robust_pred': False}]

display_violin_2 = [{'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                    {'strategy': 'dro', 'sample': 'test', 'radius': 5e-7, 'robust_pred': False},

                    {'strategy': 'dro', 'sample': 'test', 'radius': 1e-7, 'robust_pred': False},

                    {'strategy': 'dro', 'sample': 'test', 'radius': 1e-9, 'robust_pred': False}]

display_violin_3 = [{'strategy': 'mv', 'sample': 'test', 'radius': 5e-6, 'robust_pred': False},

                    {'strategy': 'dro', 'sample': 'test', 'radius': 0.1, 'robust_pred': False},

                    {'strategy': 'dro', 'sample': 'test', 'radius': 1, 'robust_pred': False},

                    {'strategy': 'dro', 'sample': 'test', 'radius': 10, 'robust_pred': False}
                    ]
