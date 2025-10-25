package com.samuelberrien.arenai.set_controls.axis;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Color;
import androidx.core.content.ContextCompat;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RelativeLayout;
import android.widget.TextView;
import com.samuelberrien.arenai.R;
import com.samuelberrien.arenai.Dimens;

import java.util.Timer;
import java.util.TimerTask;

public class ContinuousInputView extends RelativeLayout implements SharedPreferences.OnSharedPreferenceChangeListener, View.OnClickListener {

	public enum OrientationEnum {

	}

	private Axis axis;
	private AxisGage axisGage;
	private TextView text;
	private boolean isPlus;
	private boolean hasClick;
	private boolean listening;
	private SharedPreferences pref;

	public ContinuousInputView(Context context, Axis axis, boolean isPlus) {
		super(context);

		this.axis = axis;
		this.isPlus = isPlus;
		listening = false;

		setBackground(ContextCompat.getDrawable(context, R.drawable.button_axis));

		axisGage = new AxisGage(context, axis, isPlus);

		pref = context.getSharedPreferences(Axis.axisPref, Context.MODE_PRIVATE);
		pref.registerOnSharedPreferenceChangeListener(this);

		text = new TextView(context);
		text.setGravity(Gravity.CENTER);
		//text.setBackground(ContextCompat.getDrawable(context, R.drawable.control_text));
		setText(pref);

		addView(axisGage, axisGage.makeLayoutParams());
		RelativeLayout.LayoutParams params = new RelativeLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
		params.addRule(RelativeLayout.CENTER_IN_PARENT);
		int px = Dimens.dpResToPx(context, R.dimen.stroke_width);
		params.setMargins(px, px, px, px);
		addView(text, params);

		text.setOnClickListener(this);
	}

	private void setText(SharedPreferences pref) {
		String key = axis.getAxisMap().getName() + (isPlus ? "+" : "-");
		text.setText(pref.getInt(key, -1) + (pref.getBoolean(key + "?", false) ? "+" : "-"));
	}

	@Override
	public void onSharedPreferenceChanged(SharedPreferences pref, String key) {
		if (key.contains(axis.getAxisMap().getName()) && key.contains(isPlus ? "+" : "-")) {
			setText(pref);
		}
	}

	@Override
	public boolean onGenericMotionEvent(MotionEvent event) {
		if (!listening)
			return super.onGenericMotionEvent(event);

		boolean handled = false;
		for (int i = MotionEvent.AXIS_X; i < MotionEvent.AXIS_GENERIC_16; i++) {
			float v = event.getAxisValue(i);
			if (v > Axis.LIMIT || v < -Axis.LIMIT) {
				String key = axis.getAxisMap().getName() + (isPlus ? "+" : "-");
				pref.edit()
						.putInt(key, i)
						.putBoolean(key + "?", v > 0.f)
						.apply();
				//text.setBackground(ContextCompat.getDrawable(getContext(), R.drawable.control_text));
				text.setBackgroundColor(Color.TRANSPARENT);
				text.requestLayout();
				listening = false;
				handled = true;
				break;
			}
		}
		return handled || super.onGenericMotionEvent(event);
	}

	@Override
	public void onClick(View v) {
		if (!listening) {
			text.setBackground(ContextCompat.getDrawable(getContext(), R.color.redTransparent));
			text.requestLayout();
			new Timer().schedule(new TimerTask() {
				@Override
				public void run() {
					post(() -> {
						//text.setBackground(ContextCompat.getDrawable(getContext(), R.drawable.control_text));
						text.setBackgroundColor(Color.TRANSPARENT);
						text.requestLayout();
						listening = false;
					});
				}
			}, 4000);
			listening = true;
			return;
		}
		listening = true;
	}
}
