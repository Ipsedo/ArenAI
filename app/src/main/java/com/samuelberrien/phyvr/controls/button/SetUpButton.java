package com.samuelberrien.phyvr.controls.button;

import android.content.Context;
import android.content.res.TypedArray;
import android.support.annotation.Nullable;
import android.support.v4.content.ContextCompat;
import android.util.AttributeSet;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.samuelberrien.phyvr.R;
import com.samuelberrien.phyvr.Dimens;

public class SetUpButton extends LinearLayout {

	private Button.ButtonMap map;
	private Button button;
	private ButtonContainer contener;

	public SetUpButton(Context context) {
		super(context);
		throw new UnsupportedOperationException("Must be initialized with button field ");
	}

	public SetUpButton(Context context, @Nullable AttributeSet attrs) {
		super(context, attrs);
		init(context.getTheme().obtainStyledAttributes(
				attrs,
				R.styleable.SetUpButton,
				0, 0), context);
	}

	public SetUpButton(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
		super(context, attrs, defStyleAttr);
		init(context.getTheme().obtainStyledAttributes(
				attrs,
				R.styleable.SetUpButton,
				defStyleAttr, 0), context);
	}

	public SetUpButton(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
		super(context, attrs, defStyleAttr, defStyleRes);
		init(context.getTheme().obtainStyledAttributes(
				attrs,
				R.styleable.SetUpButton,
				defStyleAttr, defStyleRes), context);
	}

	private void init(TypedArray a, Context context) {

		try {
			switch (a.getInteger(R.styleable.SetUpButton_button, -1)) {
				case 10:
					map = Button.ButtonMap.BRAKE;
					break;
				case 11:
					map = Button.ButtonMap.RESPAWN;
					break;
				case 12:
					map = Button.ButtonMap.FIRE;
					break;
			}
		} catch (RuntimeException uoe) {
			uoe.printStackTrace();
		} finally {
			a.recycle();
		}

		button = new Button(context, map);
		setOrientation(VERTICAL);

		TextView name = new TextView(context);
		name.setGravity(Gravity.CENTER);
		name.setBackground(ContextCompat.getDrawable(context, R.color.greyTransparent));
		name.setText(map.getFullName());

		LinearLayout.LayoutParams params = new LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
		params.weight = 0.6f;
		int px = Dimens.dpResToPx(context, R.dimen.stroke_width);
		params.setMargins(px, px, px, 0);

		addView(name, params);

		params = new LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
		params.weight = 0.4f;

		contener = new ButtonContainer(context, button);

		addView(contener, params);
	}

	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		return contener.onKeyDown(keyCode, event) || super.onKeyDown(keyCode, event);
	}

	@Override
	public boolean onKeyUp(int keyCode, KeyEvent event) {
		return contener.onKeyUp(keyCode, event) || super.onKeyDown(keyCode, event);
	}
}
