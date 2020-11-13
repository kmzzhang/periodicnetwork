import numpy as np


class LightCurve:
    def __init__(self, times, measurements, errors=None, survey=None, name=None,
                 best_period=None, best_score=None, label=None, p=None,
                 p_signif=None, p_class=None, ss_resid=None, metadata=None):
        """

        Parameters
        ----------
        times: ndarray
            1D array of shape (L,)
        measurements: ndarray
            1D array of shape (L,)
        errors: ndarray
            (optional) 1D array of shape (L,)
        survey: string
            survey name
        name: string
            object name
        best_period: float
            obsolete. load period into p instead
        best_score: float
            obsolete
        label: string or int
            class of light-curve. can be name or number.
        p: float
            period
        p_signif: float
            obsolete
        p_class: float
            obsolete
        ss_resid: float
            obsolete
        metadata: ndarray
            1D array of features from external catelogs to be used as auxiliary network inputs
        """
        self.times = times
        self.measurements = measurements
        self.errors = errors if errors is not None else np.zeros_like(times)
        self.label = label
        self.metadata = metadata
        self.survey = survey
        self.name = name

        # period
        self.p = p

        # optional
        self.best_period = best_period
        self.best_score = best_score
        self.class_prob = None
        self.ss_resid = ss_resid
        self.p_signif = p_signif
        self.p_class = p_class


    def __repr__(self):
        return "LightCurve(" + ', '.join("{}={}".format(k, v)
                        for k, v in self.__dict__.items()) + ")"

    def __len__(self):
        return len(self.times)

    def split(self, n_min=0, n_max=np.inf):
        inds = np.arange(len(self.times))
        splits = [np.array(x)
                  for x in np.array_split(inds, np.arange(n_max, len(inds), step=n_max))
                  if len(x) >= n_min]
        return [LightCurve(survey=self.survey, name=self.name,
                           times=self.times[s],
                           measurements=self.measurements[s],
                           errors=self.errors[s], best_period=self.best_period,
                           best_score=self.best_score, label=self.label,
                           p=self.p, p_signif=self.p_signif, p_class=self.p_class,
                           ss_resid=self.ss_resid)
                for s in splits]

    def fit_supersmoother(self, periodic=True, scale=True):
        from supersmoother import SuperSmoother
        model = SuperSmoother(period=self.p if periodic else None)
        try:
            model.fit(self.times, self.measurements, self.errors)
            self.ss_resid = np.sqrt(np.mean((model.predict(self.times) - self.measurements) ** 2))
            if scale:
                self.ss_resid /= np.std(self.measurements)
        except ValueError:
            self.ss_resid = np.inf

    def period_fold(self, p=None):
        self.times_copy = np.copy(self.times)
        self.measurements_copy = np.copy(self.measurements)
        self.errors_copy = np.copy(self.errors)
        if p is None:
            p = self.p
        self.times = self.times % p
        inds = np.argsort(self.times)
        self.times = self.times[inds]
        self.measurements = self.measurements[inds]
        self.errors = self.errors[inds]
        self.inds = inds

    def period_unfold(self):
        self.times = self.times_copy
        self.measurements = self.measurements_copy
        self.errors = self.errors_copy
