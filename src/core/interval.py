
# @brief Representation of an interval in R that can be bounded or unbounded
# and closed or open on both ends. Also supports sampling and predicates such
# as compactness and containment.
class Interval:
    # ===================
    # Private Members
    # ===================
    __m_t0: float
    __m_t1: float
    __m_bounded_below: bool
    __m_bounded_above: bool
    __m_open_below: bool
    __m_open_above: bool

    def __init__(self):
        self.reset_bounds()

    def __init__(self, lower_bound: float, upper_bound: float):
        set_lower_bound(lower_bound)
        set_upper_bound(upper_bound)

    def set_lower_bound(lower_bound: float,  is_open: bool = False):

    def set_upper_bound(upper_bound: float, is_open: bool = False):

    def trim_lower_bound(trim_amount: float):
    def trim_upper_bound(trim_amount: float):
    def pad_lower_bound(pad_amount: float):
    def pad_upper_bound(pad_amount: float):

    @property
    def get_lower_bound() -> float:
        return m_t0

    def get_upper_bound() -> float:
    def get_center() -> float:
    def get_length() -> float:

    @property
    bool is_bounded_below() const

    @property
    bool is_bounded_above() const

    @property
    bool is_open_below() const

    @property
    bool is_open_above() const

    @property
    bool is_finite() const

    @property
    bool is_compact() const

    # TODO: float("inf") may be wrong...
    def reset_bounds(self) -> None:
        m_t0 = -float("inf")
        m_t1 = float("inf")
        m_bounded_below = False
        m_bounded_above = False
        m_open_below = True
        m_open_above = True

    @property
    bool contains(self, t: float) const

    @property
    bool is_in_interior(double t) const

    std:: vector < double > sample_points(int num_points) const

    std: : string formatted_interval() const
