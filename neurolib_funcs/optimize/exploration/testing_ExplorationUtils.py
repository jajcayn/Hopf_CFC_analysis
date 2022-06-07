import os
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

import logging
import tqdm

from scipy import stats

from neurolib_funcs.utils import paths as paths




def plotExplorationResults(
        dfResults,
        par1,
        par2,
        plot_key,
        p_vals_label,
        nan_to_zero=False,
        by=None,
        by_label=None,
        plot_key_label=None,
        symmetric_colorbar=False,
        one_figure=False,
        contour=None,
        alpha_mask=None,
        multiply_axis=None,
        savename=None,
        p_vals=None,

        **kwargs,
):
    """
    """
    # PREPARE DATA
    # ------------------
    # copy here, because we add another column that we do not want to keep later
    dfResults = dfResults.copy()

    if isinstance(par1, str):
        par1_label = par1
    elif isinstance(par1, (list, tuple)):
        par1_label = par1[1]
        par1 = par1[0]

    if isinstance(par2, str):
        par2_label = par2
    elif isinstance(par2, (list, tuple)):
        par2_label = par2[1]
        par2 = par2[0]

    # the `by` argument is used to slice the data
    # if no by value was given, create a dummy value
    if by is None:
        dfResults["_by"] = 0
        by = ["_by"]

    if by_label is None:
        by_label = by

    n_plots = len(dfResults.groupby(by=by))

    # PLOT
    # ------------------

    # create subfigures
    if one_figure == True:
        fig, axs = plt.subplots(nrows=1, ncols=n_plots, figsize=(n_plots * 4, 3.5), dpi=150)
        if plot_key_label:
            fig.suptitle(plot_key_label)

    axi = 0
    # cycle through all slices
    for i, df in dfResults.groupby(by=by):
        # chose the current axis
        if one_figure == True:
            if n_plots > 1:
                ax = axs[axi]
            else:
                ax = axs
        else:
            fig = plt.figure(figsize=(5, 4), dpi=150)
            if plot_key_label:
                plt.title(plot_key_label)
            ax = plt.gca()

        # -----
        # pivot data and plot
        df_pivot = df.pivot_table(values=plot_key, index=par2, columns=par1, dropna=False,aggfunc=np.nanmean)



        if nan_to_zero:
            df_pivot = df_pivot.fillna(-np.inf)

        plot_clim = kwargs.get("plot_clim", (np.nanmin(df_pivot.values), np.nanmax(df_pivot.values)))

        if symmetric_colorbar:
            plot_clim = (-np.max(np.abs(plot_clim)), np.max(np.abs(plot_clim)))
        image_extent = [min(df[par1]), max(df[par1]), min(df[par2]), max(df[par2])]
        image = np.array(df_pivot)

        # -----
        # alpha mask
        if alpha_mask is not None:
            mask_threshold = kwargs.get("mask_threshold", 1)
            mask_alpha = kwargs.get("mask_alpha", 0.5)
            mask_style = kwargs.get("mask_style", None)
            mask_invert = kwargs.get("mask_invert", False)

            # alpha_mask can either be a pd.DataFrame or an np.ndarray that is
            # layed over the image, a string that is a key in the results df
            # or simply True, which means that the image itself will be used as a
            # threshold for the alpha map (default in alphaMask()).

            if isinstance(alpha_mask, (pd.DataFrame, np.ndarray)):
                mask = np.array(alpha_mask)
            elif alpha_mask == "custom":
                mask = df.pivot_table(values=alpha_mask, index=par2, columns=par1, dropna=False)
            elif isinstance(alpha_mask, str):
                mask = df.pivot_table(values=alpha_mask, index=par2, columns=par1, dropna=False,aggfunc=np.nanmean)
                if nan_to_zero:
                    mask = mask.fillna(0)
                mask = np.array(mask)
            else:
                mask = None

            image = alphaMask(image, mask_threshold, mask_alpha, mask=mask, invert=mask_invert, style=mask_style, )

        im = ax.imshow(image, extent=image_extent, origin="lower", aspect="auto", clim=plot_clim, )

        # p-values
        if p_vals is not None:

            df_p_vals = df.pivot_table(values=p_vals_label, index=par2, columns=par1, aggfunc=np.nanmean)
            df_p_vals = df_p_vals.fillna(-np.inf)

            df_p_vals_binary = (np.array((df_p_vals < 0.1) & (df_p_vals >= 0)).astype(int))


            X_pvals, Y_pvals = np.meshgrid(np.unique(df[par2]),np.unique(df[par1]))
            # offset_X = np.unique(np.diff(df[par1]))[1]/4
            # offset_Y = np.unique(np.diff(df[par2]))[1]/4
            offset_X = 0
            offset_Y = 0
            X_pvals = X_pvals + offset_X
            Y_pvals = Y_pvals + offset_Y


            ax.scatter(X_pvals, Y_pvals, c=df_p_vals_binary, marker="x", cmap="PiYG", clim=plot_clim,s=20)


        # ANNOTATIONs
        # ------------------
        # plot contours
        if contour is not None:
            contour_color = kwargs.get("contour_color", "white")
            contour_levels = kwargs.get("contour_levels", None)
            contour_alpha = kwargs.get("contour_alpha", 1)
            contour_kwargs = kwargs.get("contour_kwargs", dict())

            def plot_contour(contour, contour_color, contour_levels, contour_alpha, contour_kwargs):
                # check if this is a dataframe
                if isinstance(contour, pd.DataFrame):
                    contourPlotDf(
                        contour,
                        color=contour_color,
                        ax=ax,
                        levels=contour_levels,
                        alpha=contour_alpha,
                        contour_kwargs=contour_kwargs,
                    )
                # if it's a string, take that value as the contour plot value
                elif isinstance(contour, str):
                    df_contour = df.pivot_table(values=contour, index=par2, columns=par1, dropna=False)
                    if nan_to_zero:
                        df_contour = df_contour.fillna(0)
                    contourPlotDf(
                        df_contour,
                        color=contour_color,
                        ax=ax,
                        levels=contour_levels,
                        alpha=contour_alpha,
                        contour_kwargs=contour_kwargs,
                    )

            # check if contour is alist of variables, e.g. ["max_output", "domfr"]
            if isinstance(contour, list):
                for ci in range(len(contour)):
                    plot_contour(
                        contour[ci], contour_color[ci], contour_levels[ci], contour_alpha[ci], contour_kwargs[ci]
                    )
            else:
                plot_contour(contour, contour_color, contour_levels, contour_alpha, contour_kwargs)

        # colorbar
        if one_figure == False:
            # useless and wrong if images don't have the same range! should be used only if plot_clim is used
            cbar = plt.colorbar(im, ax=ax, orientation="vertical", label=plot_key_label)
        else:
            # colorbar per plot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

        # labels and style
        ax.set_xlabel(par1_label)
        ax.set_ylabel(par2_label)

        # tick marks
        ax.tick_params(
            axis="both", direction="out", length=3, width=1, bottom=True, left=True,
        )

        # multiply / rescale axis
        if multiply_axis:
            ax.set_xticklabels(np.round(np.multiply(ax.get_xticks(), multiply_axis), 2))
            ax.set_yticklabels(np.round(np.multiply(ax.get_yticks(), multiply_axis), 2))

        # single by-values need to become tuple
        if not isinstance(i, tuple):
            i = (i,)
        if by != ["_by"]:
            title = "-".join([f"{bb}={bi}" for bb, bi in zip(by_label, i)])
            ax.set_title(title)
        if one_figure == False:
            if savename:
                save_fname = os.path.join(paths.FIGURES_DIR, f"{title}_{savename}")
                plt.savefig(save_fname)
                logging.info(f"Saving to {save_fname}")
            plt.show()
        else:
            axi += 1

    if one_figure == True:
        plt.tight_layout()
        if savename:
            save_fname = os.path.join(paths.FIGURES_DIR, f"{savename}")
            plt.savefig(save_fname)
            logging.info(f"Saving to {save_fname}")
        plt.show()



def contourPlotDf(
        dataframe,
        color="white",
        levels=None,
        ax=None,
        alpha=1.0,
        countour=True,
        contourf=False,
        clabel=False,
        **contour_kwargs,
):
    levels = levels or [0, 1]
    Xi, Yi = np.meshgrid(dataframe.columns, dataframe.index)
    ax = ax or plt

    if contourf:
        contours = plt.contourf(Xi, Yi, dataframe, 10, levels=levels, alpha=alpha, cmap="plasma")

    # unpack, why necessary??
    contour_kwargs = contour_kwargs["contour_kwargs"]

    contours = ax.contour(Xi, Yi, dataframe, colors=color, levels=levels, zorder=1, alpha=alpha, **contour_kwargs, )

    clabel = contour_kwargs["clabel"] if "clabel" in contour_kwargs else False
    if clabel:
        ax.clabel(contours, inline=True, fontsize=8)


def alphaMask(image, threshold, alpha, mask=None, invert=False, style=None):
    """Create an alpha mask on an image using a threshold

    :param image: RGB image to create a mask on.
    :type image: np.array (NxNx3)
    :param threshold: Threshold value
    :type threshold: float
    :param alpha: Alpha value of mask
    :type alpha: float
    :param mask: A predefined mask that can be used instead of the image itself, defaults to None
    :type mask: np.array, optional
    :param invert: Invert the mask, defaults to False
    :type invert: bool, optional
    :param style: Chose a style for the mask, currently only `stripes` supported, defaults to None
    :type style: string, optional
    :return: Masked image (RGBA), 4-dimensional (NxNx4)
    :rtype: np.array
    """
    if mask is None:
        mask = image

    alphas = mask > threshold if not invert else mask < threshold
    alphas = np.clip(alphas, alpha, 1)

    if style == "stripes":
        f = mask.shape[0] / 5
        style_mask = np.sin(np.linspace(0, 2 * np.pi * f, mask.shape[0]))
        style_mask = style_mask > 0
        alphas = alphas + style_mask[:, None]
        alphas = np.clip(alphas, 0, 1)

    cmap = plt.cm.plasma
    colors = Normalize(np.nanmin(image), np.nanmax(image), clip=True)(image)
    colors = cmap(colors)
    colors[..., -1] = alphas
    return colors


def plotResult(search, runId, z_bold=False, **kwargs):
    fig, axs = plt.subplots(1, 3, figsize=(8, 2), dpi=300, gridspec_kw={"width_ratios": [1, 1.2, 2]})

    bold_transient = int(kwargs["bold_transient"] / 2) if "bold_transient" in kwargs else int(30 / 2)

    # get result from search
    result = search.getResult(runId)

    bold = result.BOLD[:, bold_transient:]
    bold_z = stats.zscore(bold, axis=1)
    t_bold = np.linspace(2, len(bold.T) * 2, len(bold.T), )

    output = result[search.model.default_output]
    output_dt = search.model.params.dt
    t_output = np.linspace(output_dt, len(output.T) * output_dt, len(output.T), )

    axs[0].set_title(f"FC (run {runId})")
    axs[0].imshow(func.fc(bold))
    axs[0].set_ylabel("Node")
    axs[0].set_xlabel("Node")

    # axs[1].set_title("BOLD")
    if z_bold:
        axs[1].plot(t_bold, bold_z.T, lw=1.5, alpha=0.8)
    else:
        axs[1].plot(t_bold, bold.T, lw=1.5, alpha=0.8)
    axs[1].set_xlabel("Time [s]")
    if z_bold:
        axs[1].set_ylabel("Normalized BOLD")
    else:
        axs[1].set_ylabel("BOLD")

    axs[2].set_ylabel("Activity")
    axs[2].plot(t_output, output.T, lw=1.5, alpha=0.6)
    axs[2].set_xlabel("Time [ms]")
    if "xlim" in kwargs:
        axs[2].set_xlim(kwargs["xlim"])
    plt.tight_layout()


