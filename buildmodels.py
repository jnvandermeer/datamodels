from pathlib import Path
import pandas as pd
from types import SimpleNamespace
from pudb.remote import set_trace
import numpy as np
# from scramblers import phase_scramble
import logging
from tqdm import tqdm
import re
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# need the following for plotting
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde


# one of the other things to take care of ,at some point, is being able to think/plot/work with different GROUPS
# but, most ofthe data iv'e seen are just 1 group.
# we'll get to that later... it would require to refactor code, but hopefully it won't be too bad
#

# we need a class that can do very simple models.
# let's just start with calling an average, a gregression, etc, models.
# it's not yet a Digital Twin Model, but perhaps this will get us un our way there
# we've already explored that Long-Format data is the thing to run on
# we've seen also already text, numbers, and mixed data
# we've seen that there are participats, there are sessions, and there are things that we measure from those
# SO. Let us check these things

# we start from similar structures as our dat ascrambler.
# These are ALL 'Stage 1' tools. Expect nothing fancy or suited for release yet.

# so,. I guess load, enterdata, and checkit, and the init, should all be in the 'csv-interface' class.
# and then we can build on top of that...
# this will be something for the next development cycle, however
# we also need time to cover everything, and time to do proper unit testing, etc.
# for now we are just developing and gaining insights as we go.

logger = logging.getLogger(__name__)



class DataLoader(object):

    # This is the smae thing as the synthesizer, working on Long Format
    # in an ()eventual) refactor, we could isolate this and make this into an 'csv-interface' object...
    # at some point..
    # I am still not sold on the 'SimpleNameSpace' thing either.
    def __init__(self):
        self.data = None
        self.params = SimpleNamespace(
            participantid=None,
            data=None,
            type=None,
            unit=None,
            cols_to_keep=None,
            session=None,
            time_value=None,
            time_unit=None,
            rep_measures=None
        )
        
        self.all_set = False
        self.history = ''
        self.errorhistory = ''

        
        # self.seed=seed
        # self.max_participants = max_participants
        # the explained varaicne that we're going to reconstruct again:
        # self.PCAthresh = PCAthresh


    def load(self, filename, **params):
        """Load a CSV and associate parameters with it."""
        self.filename = filename
        self.dataframe = pd.read_csv(filename)

        for key, value in params.items():
            setattr(self.params, key, value)

        # we can also immediately make the empty dataframe.
        self.output = self.dataframe.reindex().astype(object)
        self.output.loc[:, :] = pd.NA

        required = ('participantid', 'session', 'data', 'type', 'unit')

        self.all_set = all(
            getattr(self.params, attr) is not None
            for attr in required
        )
        
        if self.all_set:
            logger.info('All column names seem to have been provided')
        else:
            logger.info('You need to enter column names with enterdata')
        
    

    def enterdata(self):
        """Check and interactively complete parameters."""

        for key, value in vars(self.params).items():

            if value is None:
                new_value = input(f'{key} is empty! Enter the new value: ')
                setattr(self.params, key, new_value)

            else:
                answer = input(
                    f'{key} is now: {value};   change it? (y/N): '
                )

                if answer.lower() == 'y':
                    new_value = input(f'Enter new value for {key}: ')
                    setattr(self.params, key, new_value)

        required = ('participantid', 'session', 'data', 'type', 'unit')

        self.all_set = all(
            getattr(self.params, attr) is not None
            for attr in required
        )

    def checkit(self):

        # once here, if all_set, we can find the things that need processing:
        if self.all_set:
    
            
            df=self.dataframe
            participants =  list(df[self.params.participantid].unique())
            print(f'There are {len(participants)} participants in this data sheet')
            for p in participants:
                print(f'\t{p}')
            sessions = sorted(list(df[self.params.session].unique()))
            print(f'There are {len(sessions)} sessions in this data sheet')
            for s in sessions:
                print(f'\t{s}')

                if pd.isna(s):
                    print('There seems to be a NA in the period/session; I will tread NA as session "NA"')
            
            type_and_unit = df[[self.params.type, self.params.unit]].drop_duplicates().values.tolist()
            print(f'There are {len(type_and_unit)} different kinds of variables/units:')
            for t, u in type_and_unit:
                print(f'\t{t} -- in units: {u}')

            if self.params.rep_measures is not None:
                print(f'Repeated measures is set, column: {self.params.rep_measures}')

            if self.params.time_value is not None:
                print(f'Time series data.')
                
            self.type_and_unit = type_and_unit
            self.participants = participants
            self.sessions = sessions
        
            
            
    
        else:
            print('Some column names are not yet set! - call enterdata')



class DataExplorer(DataLoader):

    

    def visualize_distributions(self, what_to_plot):

        fig_save_dir = Path.cwd() / 'Figures'
        fig_save_dir.mkdir(exist_ok=True)

        

        df = self.dataframe

        type_and_unit = self.type_and_unit # yeah even though we COULD have 1 type and different units - we now assume each type is UNIQUE

        # a way to get the type and unit out of everything...
        # i think there might be more optimal ways for this bad programming.
        [t, u] = [[x, y] for x, y in type_and_unit if x == what_to_plot][0]

        print(f'Doing Now Images for: {t} ({u})')
        my_slice = \
        (df[self.params.type]==t) & \
        (df[self.params.unit]==u)


        # this will average over anything else - e.h., repeats and the like.
        #  my_detail is the thing we're going to be using to plot stuff!
        # from this - we may also need to plot the differences.
        my_detail = df[my_slice].groupby([self.params.session,self.params.type,self.params.unit, self.params.participantid]).agg('mean').reset_index()

        # we build the save filename already...
        my_save_fig_filename = str(fig_save_dir / re.sub('/','_', f'Distributions_{t}_{u}')) + '.png'


        my_fig = _make_figure(self, my_detail, 'Distributions')
        my_fig.write_image(my_save_fig_filename)
        print(f'saved: {my_save_fig_filename}')



    def visualize_differences(self, what_to_plot):

        fig_save_dir = Path.cwd() / 'Figures'
        fig_save_dir.mkdir(exist_ok=True)


        

        df = self.dataframe

        type_and_unit = self.type_and_unit # yeah even though we COULD have 1 type and different units - we now assume each type is UNIQUE

        # a way to get the type and unit out of everything...
        # i think there might be more optimal ways for this bad programming.
        [t, u] = [[x, y] for x, y in type_and_unit if x == what_to_plot][0]

        print(f'Doing Now Images for: {t} ({u})')
        my_slice = \
        (df[self.params.type]==t) & \
        (df[self.params.unit]==u)


        # this will average over anything else - e.h., repeats and the like.
        #  my_detail is the thing we're going to be using to plot stuff!
        # from this - we may also need to plot the differences.
        my_detail = (
            df[my_slice]
            .groupby([
                self.params.session,
                self.params.type,
                self.params.unit,
                self.params.participantid,
            ])
            .agg('mean')
            .reset_index()
        )

        # print(sessions)

        # Calculate differences -- pivot to matrix -> diff -> back to long format -> update stuff
        df_wide = my_detail.pivot(
            index=[self.params.type, self.params.unit, self.params.participantid],
            columns=self.params.session,
            values=self.params.data
        )

        diff = df_wide[self.sessions].diff(axis=1).iloc[:, 1:]
        diff.columns = [
            f'{current} - {previous}'
            for previous, current in zip(self.sessions[:-1], self.sessions[1:])
        ]
        
        # return to long format again:
        df_diff = (
            diff.rename_axis(columns=self.params.session)
                .stack()
                .rename(self.params.data)
                .reset_index()
        )
        
        
        mean_repeat = (
            my_detail.groupby(self.params.participantid)['repeat']
            .mean()
            .rename('repeat')
        )
        df_diff = df_diff.merge(mean_repeat, on=self.params.participantid, how='left')
        


        # we build the save filename already...
        my_save_fig_filename = str(fig_save_dir / re.sub('/','_', f'Differences_{t}_{u}')) + '.png'

        my_fig = _make_figure(self, df_diff, 'Differences', do_hline=1)
        my_fig.write_image(my_save_fig_filename)
        print(f'saved: {my_save_fig_filename}')


    

    

# so, depending on what my_detail is - we can plot anything...
# we DO cheat a little bit... by passing the object along...
# But, this will allow us to make NAOTHER my_detail, e.g. with less number of sessions (-1), containing the differences
# nice thing is that we can put these functions anywhere...

def _make_figure(self, my_detail, title_begin, do_hline=0):


    sessions=self.sessions
    participants=self.participants

    
    # we should already have called checkit and load by now, so things are filled in.
    # Identical jitter for every participant at each session
    rng = np.random.default_rng(42)

    jitter = dict(zip(participants, rng.uniform(-0.08, 0.08, len(participants))))
    session_pos = {s: i for i, s in enumerate(sorted(my_detail[self.params.session].unique()))}

    # we need xpos added:
    my_detail["x_pos"] = my_detail[self.params.session].map(session_pos) + my_detail[self.params.participantid].map(jitter)

    # Light blue -> dark blue, with reasonably visible differences
    colors = px.colors.sample_colorscale(
        "Blues",
        [0.25 + 0.65 * i / max(len(participants) - 1, 1)
         for i in range(len(participants))]
    )
    color_map = dict(zip(participants, colors))

    fig = go.Figure()
    participant_traces = {}


    for participant, d in my_detail.groupby(self.params.participantid):

        color = color_map[participant]

        participant_traces[participant] = {}

        
        participant_traces[participant]["line"] = len(fig.data)


        # we add LINES
        fig.add_trace(go.Scatter(
            x=d["x_pos"],
            y=d[self.params.data],
            mode="lines",
            opacity=0.2,
            showlegend=False,
            line=dict(width=1.5, color=color),
            legendgroup=str(participant),
            name=str(participant),
            customdata=[str(participant)] * len(d),
            hovertemplate="Participant: %{customdata}<extra></extra>",
        ))


        # we add POINTS
        # Participant markers
        participant_traces[participant]["markers"] = len(fig.data)
        fig.add_trace(go.Scatter(
            x=d["x_pos"],
            y=d[self.params.data],
            mode="markers",
            opacity=1,
            showlegend=False,
            marker=dict(size=5, color="black", opacity=1),
            legendgroup=str(participant),
            name=str(participant),
            customdata=[str(participant)] * len(d),
            hovertemplate="Participant: %{customdata}<br>Value: %{y}<extra></extra>",
        ))


    # we add the BOXES - i.e. the boxplots
    for trial, d in my_detail.groupby(self.params.session):
        pos = session_pos[trial]

        fig.add_trace(go.Box(
            x=[pos - 0.18] * len(d),
            y=d[self.params.data],
            width=0.13,
            boxpoints=False,
            quartilemethod="linear",
            whiskerwidth=0,
            # fillcolor="rgba(255,255,255,0.65)",
            # fillcolor="rgba(255,255,255,0.35)",
            # fillcolor="rgba(190, 190, 190, 0.55)",
            fillcolor="rgba(90, 90, 90, 0.75)",
            line=dict(color="black", width=1.5),
            marker=dict(color="black", size=5),
            showlegend=False,
            name=f"{trial} box",
            zorder=20,
        ))


    # we add the VIOLINS!
    for trial, d in my_detail.groupby(self.params.session):
        pos = session_pos[trial]
    
        fig.add_trace(go.Violin(
            x=[pos + 0.15] * len(d),
            y=d[self.params.data],
            side="positive",
            width=0.6,
            points=False,
            box_visible=False,
            meanline_visible=False,
            fillcolor="rgba(90, 90, 90, 0.75)",
            line=dict(width=1.2, color="black"),
            opacity=1,
            showlegend=False,
            name=str(trial),
        ))


        # Added VERY DODGY chatgpt code to make the 1 line...
        # Estimate the KDE used by the violin
        values = d[self.params.data].dropna().to_numpy()
        kde = gaussian_kde(values)
        
        y_grid = np.linspace(values.min() - values.std(),
                             values.max() + values.std(), 1000)
        density = kde(y_grid)
        
        # Approximate the visible KDE endpoints
        visible = density > density.max() * 0.01
        y_min = y_grid[visible][0]
        y_max = y_grid[visible][-1]
        
        # Draw vertical black line at the centre of the half-violin
        x_center = pos + 0.15
        
        fig.add_trace(go.Scatter(
            x=[x_center, x_center],
            y=[y_min, y_max],
            mode="lines",
            line=dict(color="black", width=1.2),
            showlegend=False,
            hoverinfo="skip",
        ))


    if do_hline:
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="black",
            line_width=1,
        )
    
        
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(
            tickmode="array",
            tickvals=list(session_pos.values()),
            ticktext=list(session_pos.keys()),
        ),
        yaxis=dict(
            title=my_detail[self.params.type].iloc[0],
        ),
    )
    
    fig.update_layout(
        margin=dict(l=20, r=5, t=40, b=20)
    )


    n_participants = my_detail[self.params.participantid].nunique()
    variable = my_detail[self.params.type].iloc[0]
    unit = my_detail[self.params.unit].iloc[0]
    
    fig.update_layout(
        title=dict(text=f"{title_begin}: <b>{variable}</b> ({unit}) — {n_participants} participants", 
        font=dict(size=24))
    )

    
    fig.update_layout(
        autosize=True,
        height=850,
        width=1400,
    )


    # instead of writing it out -- lets try making this into a widget:
    from IPython.display import display
    
    # Requires a Jupyter environment with ipywidgets support
    fig_widget = go.FigureWidget(fig)
    
    
    
    hovered_participant = None
    
    def highlight_participant(trace, points, state):
        global hovered_participant
    
        if not points.point_inds:
            return
    
        participant = str(trace.customdata[points.point_inds[0]])
    
        if participant == hovered_participant:
            return
    
        hovered_participant = participant
    
        with fig_widget.batch_update():
            for p, traces in participant_traces.items():
                line = fig_widget.data[traces["line"]]
                markers = fig_widget.data[traces["markers"]]
    
                if str(p) == participant:
                    line.opacity = 1
                    line.line.width = 3
                    markers.marker.size = 9
                else:
                    line.opacity = 0.06
                    line.line.width = 1
                    markers.marker.size = 4
    
    
    def reset_highlight(trace, points, state):
        global hovered_participant
    
        hovered_participant = None
    
        with fig_widget.batch_update():
            for p, traces in participant_traces.items():
                fig_widget.data[traces["line"]].opacity = 0.2
                fig_widget.data[traces["line"]].line.width = 1.5
                fig_widget.data[traces["markers"]].marker.size = 5
    
    
    for p, traces in participant_traces.items():
        marker_trace = fig_widget.data[traces["markers"]]
        marker_trace.on_hover(highlight_participant)
        marker_trace.on_unhover(reset_highlight)
    
    display(fig_widget)

    return fig



    # in order to plot differences - we need to think how to restructure the function above. We ideally need to make a matrix
    # OR - a new dataframe of sorts, and work our way from there.
    