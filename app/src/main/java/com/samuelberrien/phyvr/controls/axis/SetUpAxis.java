package com.samuelberrien.phyvr.controls.axis;

import android.content.Context;
import android.content.res.TypedArray;
import android.support.annotation.Nullable;
import android.support.v4.content.ContextCompat;
import android.util.AttributeSet;
import android.view.*;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.samuelberrien.phyvr.R;

import static com.samuelberrien.phyvr.controls.axis.Axis.AxisMap.*;

public class SetUpAxis extends LinearLayout {

	private Axis.AxisMap axisMap;
	private Axis axis;

	public SetUpAxis(Context context) {
		super(context);
	}

	public SetUpAxis(Context context, @Nullable AttributeSet attrs) {
		super(context, attrs);
		init(context, attrs);
	}

	public SetUpAxis(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
		super(context, attrs, defStyleAttr);
		init(context, attrs);
	}

	public SetUpAxis(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
		super(context, attrs, defStyleAttr, defStyleRes);
		init(context, attrs);
	}

	private void init(Context context, AttributeSet attrs) {
		TypedArray a = context.getTheme().obtainStyledAttributes(
				attrs,
				R.styleable.SetUpAxis,
				0, 0);

		try {
			switch (a.getInteger(R.styleable.SetUpAxis_command, -1)) {
				case 0: axisMap = DIR; break;
				case 1: axisMap = SPEED; break;
				case 2: axisMap = CANON; break;
				case 3: axisMap = TURRET; break;
			}
		} catch (RuntimeException uoe) {
			uoe.printStackTrace();
		} finally {
			a.recycle();
		}

		System.out.println("TTTTTTTTTTTTTTTTTT " + axisMap.getFullName());

		axis = new Axis(getContext(), axisMap);

		setOrientation(VERTICAL);

		LinearLayout l = new LinearLayout(context);
		l.setOrientation(HORIZONTAL);
		LinearLayout.LayoutParams params = new LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
		params.weight = 0.5f;
		l.addView(new AxisContener(context, axis, false), params);
		l.addView(new AxisContener(context, axis, true), params);

		TextView name = new TextView(context);
		name.setBackground(ContextCompat.getDrawable(getContext(), R.drawable.button_axis));
		name.setGravity(Gravity.CENTER);
		name.setText(axisMap.getFullName());

		params = new LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
		params.weight = 0.6f;
		addView(name, params);

		params = new LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
		params.weight = 0.4f;
		addView(l, params);
	}

	@Override
	public boolean onGenericMotionEvent(MotionEvent event) {
		axis.onGenericMotion(event);
		return super.onGenericMotionEvent(event);
	}
}
