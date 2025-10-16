package com.samuelberrien.phyvr.set_controls.axis;

import android.content.Context;
import androidx.core.content.ContextCompat;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.RelativeLayout;
import com.samuelberrien.phyvr.R;
import com.samuelberrien.phyvr.Dimens;

public class AxisGage extends View implements Axis.OnAxisMoveListener {

	private Axis axis;
	private boolean isPlus;
	private float size;

	public AxisGage(Context context, Axis axis, boolean isPlus) {
		super(context);

		this.isPlus = isPlus;
		this.axis = axis;
		this.axis.addListener(this);

		setBackground(ContextCompat.getDrawable(context, R.color.colorAccent));

		ViewTreeObserver viewTreeObserver = getViewTreeObserver();
		if (viewTreeObserver.isAlive()) {
			viewTreeObserver.addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
				@Override
				public void onGlobalLayout() {
					getViewTreeObserver().removeOnGlobalLayoutListener(this);
					size = getWidth();
					getLayoutParams().width = 0;
					requestLayout();
				}
			});
		}
	}

	public RelativeLayout.LayoutParams makeLayoutParams() {
		RelativeLayout.LayoutParams layoutParams = new RelativeLayout.LayoutParams(
				RelativeLayout.LayoutParams.MATCH_PARENT, RelativeLayout.LayoutParams.MATCH_PARENT);
		int px = Dimens.dpResToPx(getContext(), R.dimen.stroke_width);
		layoutParams.setMargins(px, px, px, px);
		layoutParams.addRule(isPlus ? RelativeLayout.ALIGN_PARENT_START : RelativeLayout.ALIGN_PARENT_END);
		layoutParams.addRule(RelativeLayout.CENTER_VERTICAL);
		return layoutParams;
	}

	@Override
	public void valueChanged(float value) {
		if (isPlus && value < 0.f || !isPlus && value > 0.f)
			return;
		getLayoutParams().width = (int) (size * Math.abs(value));
		requestLayout();
	}
}
