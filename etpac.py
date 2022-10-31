import numba
import numpy as np
from numba import float32, float64, guvectorize, vectorize
import xarray as xr
import matplotlib.pyplot as plt

eddy_kinds = {
    # "all": {"color": "gray", "lat0": [5, 17], "lon0": [360 - 110, 275]},
    "teh": {"color": "r", "lat0": [13, 15.5], "lon0": [260, 275]},
    "pap": {"color": "b", "lat0": [-np.inf, 13], "lon0": [360 - 90, 275]},
    "oce": {"color": "g", "lat0": [8, 12], "lon0": [360 - 105, 360 - 97]},
    #"pan": {"color": "teal", "lat0": [0, 8], "lon0": [260, 290]},  # detection prevented by amplitude criteria
}


@guvectorize(
    [
        (float32[:], float64[:], float64[:]),
        (float64[:], float64[:], float64[:]),
    ],
    "(i), (i) -> ()",
    nopython=True,
)
def _gufunc_hmxl(b, z, out):
    out[:] = np.nan

    mask = ~np.isnan(b)
    b = b[mask]
    z = z[mask]

    if len(b) == 0:
        return

    max_dbdz_surf = ((b - b[0]) / z).max()

    dbdz = np.diff(b) / np.diff(z)

    for idx in range(len(dbdz)):
        if dbdz[idx] > max_dbdz_surf:
            break

    ip = idx + 1
    if dbdz[idx] - dbdz[ip] < 0:
        ip = idx - 1

    if np.abs(dbdz[idx] - dbdz[ip]) < 1e-10:
        hmxl = z[ip]
    else:
        hmxl = z[ip] + np.abs(
            (z[idx] - z[ip]) / (dbdz[idx] - dbdz[ip]) * (max_dbdz_surf - dbdz[ip])
        )

    out[:] = hmxl


def calc_hmxl(pdens):
    z = pdens.cf["vertical"]
    zdim = z.name
    return xr.apply_ufunc(
        _gufunc_hmxl,
        -9.81 / 1025 * pdens,
        -np.abs(z),
        input_core_dims=[(zdim,), (zdim,)],
        dask="parallelized",
        output_dtypes=[z.dtype],
    )


def pdens(S, theta):
    """ Wright 97 EOS from https://mom6-analysiscookbook.readthedocs.io/en/latest/05_Buoyancy_Geostrophic_shear.html """

    @vectorize(["float32(float32, float32)"])
    def eos(S, theta):
        # --- Define constants (Table 1 Column 4, Wright 1997, J. Ocean Tech.)---
        a0 = 7.057924e-4
        a1 = 3.480336e-7
        a2 = -1.112733e-7

        b0 = 5.790749e8
        b1 = 3.516535e6
        b2 = -4.002714e4
        b3 = 2.084372e2
        b4 = 5.944068e5
        b5 = -9.643486e3

        c0 = 1.704853e5
        c1 = 7.904722e2
        c2 = -7.984422
        c3 = 5.140652e-2
        c4 = -2.302158e2
        c5 = -3.079464

        # To compute potential density keep pressure p = 100 kpa
        # S in standard salinity units psu, theta in DegC, p in pascals

        p = 100000.0
        alpha0 = a0 + a1 * theta + a2 * S
        p0 = (
            b0
            + b1 * theta
            + b2 * theta ** 2
            + b3 * theta ** 3
            + b4 * S
            + b5 * theta * S
        )
        lambd = (
            c0
            + c1 * theta
            + c2 * theta ** 2
            + c3 * theta ** 3
            + c4 * S
            + c5 * theta * S
        )

        pot_dens = (p + p0) / (lambd + alpha0 * (p + p0))

        return pot_dens

    return xr.apply_ufunc(eos, S, theta, dask="parallelized", output_dtypes=[S.dtype])


def calc_mld(pden):

    drho = pden - pden.cf.isel(Z=0)
    return xr.where(drho > 0.015, np.abs(drho.cf["Z"]), np.nan).cf.min("Z")


def annotate_stats(ax, data, x=0.95, y=0.95, va="top", ha="right", **kwargs):
    """ Annotates axes with statistics of the plotted array. Only works with pcolormesh."""
    text = f"Mean: {np.abs(data).mean().values:.2f}\nMax: {data.max().values:.2f}\nMin: {data.min().values:.2f}"
    ax.text(x=x, y=y, s=text, transform=ax.transAxes, ha=ha, va=va, **kwargs)


def contour_over_under(fg, levels=3, **kwargs):
    assert isinstance(levels, int)

    minimum = fg.data.min().values
    maximum = fg.data.max().values

    norm = fg._mappables[0].norm
    vmin = norm.vmin
    vmax = norm.vmax

    delta = min(abs(vmin - minimum), abs(vmax - maximum))
    dlevels = np.linspace(0.1, 0.9, levels) * delta
    newlevels = sorted(np.concatenate([vmin - dlevels, vmax + dlevels]))

    kwargs.setdefault("add_colorbar", False)
    kwargs.setdefault("colors", "w")
    kwargs.setdefault("linewidths", 1)

    fg.map_dataarray(
        xr.plot.contour,
        x=fg._x_var,
        y=fg._y_var,
        levels=newlevels,
        **kwargs,
    )


def plot_costa_rica_dome_extent():
    plt.plot(
        [360 - 95, 360 - 85, 360 - 85, 360 - 95, 360 - 95], [6, 6, 11, 11, 6], color="k"
    )


def fg_map(fg, func, *args, **kwargs):
    for ax, loc in zip(fg.axes.flat, fg.name_dicts.flat):
        subset = fg.data.loc[loc]
        func(ax, subset, *args, **kwargs)


def single_contour_over_under(array, hdl, levels=3, **kwargs):
    assert isinstance(levels, int)

    data = array

    minimum = data.min().values
    maximum = data.max().values

    norm = hdl.norm
    vmin = norm.vmin
    vmax = norm.vmax

    delta = min(abs(vmin - minimum), abs(vmax - maximum))
    dlevels = np.linspace(0.1, 0.9, levels) * delta
    newlevels = sorted(np.concatenate([vmin - dlevels, vmax + dlevels]))

    kwargs.setdefault("add_colorbar", False)
    kwargs.setdefault("colors", "w")
    kwargs.setdefault("linewidths", 1)

    array.plot.contour(levels=newlevels, ax=hdl.axes, **kwargs)


def euc_transport_z(u):
    u = u.cf.sel(latitude=slice(-3, 3))
    trans = u.where(u > 0).fillna(0).cf.integrate("latitude") * 105e3
    trans.attrs["units"] = "m²/s"
    trans.attrs["long_name"] = "EUC transport [$\int$ u {u > 0} dy]"
    return trans


def euc_transport_y(u):
    u = u.cf.sel(vertical=slice(500))
    trans = u.where(u > 0).fillna(0).cf.integrate("vertical")
    trans.attrs["units"] = "m²/s"
    trans.attrs["long_name"] = "EUC transport [$\int$ u {u > 0} dz]"
    return trans


def plot_section(lon):
    def subsetter(ds):
        return ds.cf.sel(longitude=360 + lon, method="nearest").cf.sel(
            latitude=slice(22)
        )

    kwargs = dict(
        ylim=(200, 0),
        levels=np.arange(10, 33, 2),
        cbar_kwargs={"orientation": "horizontal"},
    )

    fg = (argo.thetao.pipe(subsetter)).cf.plot.contour(
        y="Z", col="time", col_wrap=4, colors="k", **kwargs
    )

    # UGLY HACK =)
    fg.data = subsetter(monthly.thetao.squeeze())
    fg.map_dataarray(xr.plot.contour, x="yh", y="zl", colors="r", **kwargs)

    for loc, ax in zip(fg.name_dicts.flat, fg.axes.flat):

        subsetter(mimocml.mld).sel(loc).plot.line(color="w", lw=4, ax=ax, _labels=False)
        hmimoc = (
            subsetter(mimocml.mld)
            .sel(loc)
            .plot.line(color="b", lw=2, ax=ax, _labels=False)
        )

        subsetter(monthly.mld).sel(loc).plot.line(color="w", lw=4, ax=ax, _labels=False)
        hmom = (
            subsetter(monthly.mld)
            .sel(loc)
            .plot.line(color="g", lw=2, ax=ax, _labels=False)
        )

    dcpy.plots.annotate_end(hmimoc[0], "MIMOC MLD", va="top")
    dcpy.plots.annotate_end(hmom[0], "MOM6 MLD")

    fg.fig.suptitle(
        f"Argo monthly mean [black] vs MOM6 monthly mean [red], {-1*lon}W", y=1.01
    )


def plot_section_diff(lon, ymax=22):
    def plot_mld(mimocmld, mom6mld, ax):
        subsetter(mimocmld).plot.line(color="w", lw=4, ax=ax, _labels=False)
        hmimoc = subsetter(mimocmld).plot.line(color="b", lw=2, ax=ax, _labels=False)

        subsetter(mom6mld).plot.line(color="w", lw=4, ax=ax, _labels=False)
        hmom = subsetter(mom6mld).plot.line(color="g", lw=2, ax=ax, _labels=False)

        return hmimoc, hmom

    def subsetter(ds):
        sub = ds.cf.sel(longitude=360 + lon, method="nearest").cf.sel(
            latitude=slice(ymax)
        )
        if "vertical" in sub.cf:
            sub = sub.cf.sel(vertical=slice(500))
        return sub

    T_kwargs = dict(
        ylim=(200, 0),
        levels=np.arange(10, 33, 2),
        cbar_kwargs={"orientation": "horizontal", "shrink": 0.8, "aspect": 40},
        add_colorbar=True,
        yincrease=False,
        colors="r",
        x="yh",
        y="zl",
    )
    dT_kwargs = dict(
        y="zl",
        ylim=(200, 0),
        robust=True,
        cmap=mpl.cm.BrBG_r,
        cbar_kwargs={
            "shrink": 0.35,
            "aspect": 20,
            "anchor": (0.0, 1),
            "label": "ΔT [°C]",
        },
        vmin=-3,
        vmax=3,
        add_colorbar=True,
    )

    annual1deg = monthly1deg.weighted(monthly1deg.time.dt.days_in_month).mean("time")
    plt.figure(constrained_layout=True)
    annual_mean_bias = subsetter(
        annual1deg.thetao.cf.interp(Z=argo.cf["Z"].values)
        - argo.thetao.weighted(monthly1deg.time.dt.days_in_month).mean("time")
    )
    annual_mean_bias.cf.plot(**dT_kwargs)
    subsetter(annual1deg).thetao.plot.contour(**T_kwargs)
    plt.title(
        f"MOM6 annual mean - Argo annual Mean [color];MOM6[red], {-1*lon}W", y=1.01
    )
    hmimoc, hmom = plot_mld(
        annual1deg.mld,
        mimocml.mld.weighted(mimocml.time.dt.days_in_month).mean("time"),
        ax=plt.gca(),
    )

    dcpy.plots.annotate_end(hmimoc[0], "MIMOC MLD", va="top")
    dcpy.plots.annotate_end(hmom[0], "MOM6 MLD")

    ####
    fg = subsetter(
        monthly1deg.thetao.chunk({"zl": -1}).interp(zl=argo.zl) - argo.thetao
    ).cf.plot(col="time", col_wrap=4, **dT_kwargs)

    # UGLY HACK =)
    fg.data = subsetter(monthly.thetao.squeeze())
    fg.map_dataarray(xr.plot.contour, **T_kwargs)

    for loc, ax in zip(fg.name_dicts.flat, fg.axes.flat):
        hmimoc, hmom = plot_mld(mimocml.mld.sel(loc), monthly.mld.sel(loc), ax=ax)

    dcpy.plots.annotate_end(hmimoc[0], "MIMOC MLD", va="top")
    dcpy.plots.annotate_end(hmom[0], "MOM6 MLD")

    fg.fig.suptitle(
        f"MOM6 monthly mean - Argo Monthly Mean [color];MOM6[red], {-1*lon}W", y=1.01
    )


def calc_eqadcp(ds):

    eqadcp = ds.cf.sel(
        latitude=0, longitude=360 - np.array([125, 110, 95]), method="nearest"
    )[["uo", "vo"]].compute()
    eqadcp["w"] = eqadcp["uo"] + 1j * eqadcp.drop("xh").rename({"xh": "xq"})["vo"]

    return eqadcp


def get_selector(ds):

    lon = ds.thetao.cf["longitude"]
    lat = ds.thetao.cf["latitude"]

    selector = {
        "longitude": slice(round(lon.min().item()), round(lon.max().item())),
        "latitude": slice(round(lat.min().item()), round(lat.max().item())),
    }

    return selector


def plot_tracks(masked, kind, initial=True, **kwargs):
    ntracks = masked.sizes["track"]
    kwargs.setdefault("lw", 0.25)
    hdl = None
    for itrack in range(masked.sizes["track"]):
        sub = masked.isel(track=itrack)
        hdl = plt.plot(
            sub.longitude,
            sub.latitude,
            label=f"{kind}: {ntracks}",
            **kwargs,
        )
        if initial:
            plt.plot(
                sub.lon0,
                sub.lat0,
                marker=".",
                color=kwargs.get("color", "k"),
                markersize=4,
            )

    return hdl