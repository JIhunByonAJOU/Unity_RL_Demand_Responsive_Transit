# DRT Drive PPO State / Reward / Penalty

## 목적

`DRTPPOVehicleDriver`는 Gley가 계산한 route를 그대로 따라가지 않고, 해당 route를 1.0m 간격의 reference path로 다시 샘플링한 뒤 PPO가 차량의 조향과 종방향 입력을 직접 제어한다.

목표는 다음과 같다.

- reference path를 따라 전진한다.
- Frenet 기준 횡방향 오차를 작게 유지한다.
- lookahead 방향과 차량 heading을 맞춘다.
- 전방 곡률을 보고 커브 전에 감속한다.
- 다른 차량과 정면/측면 안전거리는 진단값으로 남긴다. 현재 reward penalty 계수는 0이다.
- 비틀거림은 차체 기준 옆속도와 좌우 반전만 약하게 penalty로 준다.
- 횡가속, 종가속, 종방향 jerk penalty는 0으로 둔다.
- 정지는 허용하되, 후진은 제한한다.

## Action

PPO action은 연속값 2개다.

```text
action[0] = steer    in [-1, 1]
action[1] = throttle in [-1, 1]
```

적용 방식:

```text
steer    -> PlayerCar steering input
throttle -> PlayerCar motor/brake input
```

`PlayerCar`는 `steer * maxSteeringAngle`을 WheelCollider steer angle로 넣는다.

## Reference Path

Gley route는 다음 API로 얻는다.

```text
API.GetPath(start, destination, vehicleType)
```

PPO driver는 raw route를 다시 샘플링한다.

```text
raw route:
  current vehicle position
  Gley waypoint positions
  destination
  destination tail point

reference path:
  raw route를 1.0m 간격으로 interpolation/resampling
```

`pathPoints`는 1.0m 간격의 reference point 목록이다. 각 reference point는 해당 위치가 속한 Gley segment의 `MaxSpeed`, `LaneWidth`, `Stop`, `GiveWay` metadata를 가진다.

reference point 통과 처리는 Frenet 진행거리 `s` 기준이다.

```text
reference waypoint spacing = 1.0m
pass distance              = 0.5m

pathPoint.s <= current_s + 0.5m 이면 통과 처리
```

기존처럼 현재 target point와의 거리만 보고 넘기지 않는다. 그래서 1m 보간 경로에서 커브 안쪽/바깥쪽 위치 때문에 waypoint가 늦게 넘어가거나 빨리 넘어가는 문제를 줄인다.

## State

현재 vector observation 크기는 77이다.

```text
10 global observations
5 lookahead reference samples * 8 = 40
9 manual rays * 3 = 27
```

### Global Observations

```text
1. current speed normalized
2. local forward velocity
3. local lateral velocity
4. yaw rate
5. destination distance
6. Frenet path progress ratio = s / total_path_length
7. cross-track error
8. heading dot
9. heading cross
10. driving flag
```

### Lookahead Reference Observations

reference path 자체는 1m 간격이지만, policy state는 바로 앞 5개 index만 보지 않는다. 현재 Frenet `s` 기준으로 다음 전방 거리들을 샘플링한다.

```text
sample distances ahead = 2m, 6m, 10m, 14m, 18m
```

이렇게 한 이유:

- 1m index 5개만 보면 policy가 최대 5m 앞까지만 보게 된다.
- 커브 진입 전 조향/감속을 배우려면 더 먼 path shape가 필요하다.
- reference path는 촘촘하게 유지하고, observation은 전방 정보를 넓게 준다.

각 sample마다:

```text
1. valid flag
2. local x
3. local z
4. distance
5. stop flag
6. giveway flag
7. Gley MaxSpeed
8. Gley LaneWidth
```

### Ray Observations

수동 ray 9개를 observation에 넣는다.

```text
angles = -80, -60, -40, -20, 0, 20, 40, 60, 80 degrees
length = 25m
```

각 ray마다:

```text
1. normalized hit distance
2. vehicle hit flag
3. non-vehicle hit flag
```

차량 근접 penalty는 ray가 아니라 collider surface distance로 계산한다.

## Path Tracking

차량 위치를 reference path segment에 projection해서 Frenet 값을 계산한다.

```text
s   = reference path 누적 진행거리
e_y = reference path에서 차량까지의 횡방향 오차
```

tracking reward의 기준 target은 가장 가까운 waypoint가 아니다.

```text
tracking target = s + dynamic_lookahead_distance 위치의 reference point
```

dynamic lookahead:

```text
lookahead = clamp(speed * 0.35s, 4m, 16m)
```

## Reward

매 fixed step마다 다음 항들을 더한다. 현재 값은 GitHub main의 주행 보상 스케일에 맞춰 되돌린 상태다.

```text
r_t =
  + path_progress_reward
  + waypoint_pass_reward
  + destination_progress_reward
  + heading_alignment_reward
  + waypoint_heading_reward
  + curve_heading_penalty
  + normalized_lateral_error_penalty
  + steering_correction_reward
  - reverse_penalty
```

### Progress Reward

```text
delta_s = max(0, s_t - s_t-1)
reward += 0.08 * delta_s
```

### Destination Progress Reward

목적지까지의 직선거리도 아주 작게 보상한다. 경로 진행 보상이 주 보상이고, 이 항목은 목적지 방향 진행을 약하게 보조한다.

```text
delta_d = max(0, previous_destination_distance - current_destination_distance)
reward += 0.002 * delta_d
```

### Waypoint Pass Reward

```text
passed_reference_point_count =
  이번 step에서 current_s + 0.5m보다 뒤에 놓이게 된 reference point 수

reward += 0.05 * passed_reference_point_count
```

main은 Gley waypoint 통과당 `0.35`였지만, 현재 코드는 route를 1.0m 간격 reference point로 재샘플링한다. 그래서 통과 보상을 그대로 쓰면 보상이 너무 촘촘하게 들어가므로 `0.05`로 낮춰 둔다.

### Tracking Shaping

횡방향 오차, heading, waypoint heading, 커브 진입 방향을 하나의 작은 dense reward로 계산한다.
`steeringCorrectionReward`는 현재 `0`이다. 한쪽 조향으로 계속 휘는 현상을 막기 위해 조향 방향 자체를 직접 보상하지 않는다.

```text
normalized_lateral_error = clamp01(e_y / maxCrossTrackErrorMeters)
steering_correction = -sign(heading_cross) * target_steering_input

tracking =
  +0.03 * heading_dot
  +0.02 * next_waypoint_heading_dot
  -0.03 * abs(heading_cross) * curve_strength
  -0.12 * normalized_lateral_error^2
  +0.00 * steering_correction * normalized_lateral_error

reward += tracking * dt
```

이전 버전처럼 `lateral_error`, `lookahead_bearing`, `lookahead_heading`을 강한 독립 penalty로 직접 빼지 않는다. 너무 강한 tracking penalty는 차가 움직일수록 손해를 보는 구조를 만들 수 있어서 main과 비슷한 작은 shaping으로 되돌렸다.

### Reverse Penalty

정지는 penalty가 없다. 후진만 penalty를 준다.

```text
if local_forward_velocity < -0.5m/s for more than 2s:
    reward += -0.5/sec * dt
```

### Vehicle Clearance Penalty

다른 차량과의 거리는 차량 중심이 아니라 collider surface 사이 최단거리로 계산한다.
현재 reward 계수 기본값은 `0`이므로 보상에는 들어가지 않는다. 진단값과 재활성화용 로직만 남겨 둔 상태다.

```text
front clearance limit = 3.0m
side clearance limit  = 1.0m
```

정면:

```text
if frontVehiclePenaltyPerSecond > 0
and front_clearance < 3.0m:
    severity = ((3.0 - front_clearance) / 3.0)^2
    reward -= frontVehiclePenaltyPerSecond * severity * dt
```

측면:

```text
if sideVehiclePenaltyPerSecond > 0
and side_clearance < 1.0m:
    severity = ((1.0 - side_clearance) / 1.0)^2
    reward -= sideVehiclePenaltyPerSecond * severity * dt
```

### Unblocked Idle Penalty

정지 자체를 항상 벌점으로 보지는 않는다. 앞차 때문에 멈춘 상황과 목적지 근처 정지는 허용한다.
현재 reward 계수 기본값은 `0`이므로 보상에는 들어가지 않는다. 무이동 종료 조건은 별도로 유지한다.

```text
idle_progress_speed =
  sum(max(0, frenet_s_delta)) over recent 1.0s / window_duration

stopped =
  idle_progress_speed < 0.1m/s

front_blocked =
  vehicle collider surface is in front corridor
  and front_clearance <= 5.0m

near_destination =
  destination_distance <= 5.0m
```

앞이 비었는데 정지해 있을 때만 grace 이후 penalty를 준다.
순간 속도 대신 최근 1초 동안의 Frenet `s` 진행량을 쓰므로, 차가 제자리에서 움찔거리거나 브레이크등이 깜빡여도 실제 경로 진행이 없으면 정지로 본다.
`current_speed < 0.2m/s`와 `local_forward_velocity < 0.2m/s`는 TensorBoard 진단값으로만 남기고, penalty reset 조건에는 직접 쓰지 않는다.

```text
if stopped and !front_blocked and !near_destination:
    idle_seconds += dt
else:
    idle_seconds = 0

if unblockedIdlePenaltyPerSecond > 0
and idle_seconds >= 1.5s:
    reward -= unblockedIdlePenaltyPerSecond * dt
```

이 항목은 초기 PPO가 "가만히 있으면 0점 근처"인 지역 최적해에 머무는 것을 막기 위한 후보였지만, 현재는 main-like reward로 되돌리기 위해 꺼 둔다.

### Overspeed Penalty

Gley waypoint의 `MaxSpeed`를 제한속도로 사용한다.
현재 reward 계수 기본값은 `0`이므로 overspeed 자체는 penalty로 들어가지 않는다. 다만 아래의 curve speed throttle shaping은 control 쪽 보정으로 유지한다.

```text
v_limit = GleyMaxSpeed / 3.6
```

전방 reference path 곡률을 본다.

```text
lookahead horizon = max(frontCurveLookaheadMeters, curvePreviewLookaheadMeters)
                  = max(20m, 30m)
sample spacing    = 2m
curvature ~= abs(delta heading) / delta s
```

곡률 기반 커브 속도:

```text
v_curve = sqrt(a_lat_limit / curvature)
```

커브까지 남은 거리와 comfortable deceleration을 반영한다.

```text
v_allowed_now = sqrt(v_curve^2 + 2 * comfortable_decel * distance_to_curve)
v_allowed = min(v_limit, v_allowed_now)
```

커브가 감지되면 곡률 강도에 따라 safety factor를 추가로 곱한다.
강한 커브에서는 `curveAllowedSpeedSafetyFactor = 0.75`까지 낮아진다.

```text
safety = lerp(1.0, 0.75, front_curve_strength)
v_allowed *= safety
```

커브 제한속도는 Gley 제한속도의 50%보다 낮게 강제하지 않는다.

```text
v_allowed >= 0.5 * v_limit
```

제한속도를 넘었을 때 penalty를 준다.
직선에서는 기존 margin `0.5m/s`를 쓰고, 커브 제한속도에서는 더 민감하게 `0.1m/s`부터 penalty를 준다.
커브 overspeed penalty는 기본 overspeed penalty의 6배다.

```text
margin = curve_limited ? 0.1m/s : 0.5m/s
multiplier = curve_limited ? 6.0 : 1.0

if overspeedPenaltyPerSecond > 0
and current_speed > v_allowed + margin:
    overspeed = current_speed - v_allowed - margin
    reward -= overspeedPenaltyPerSecond * multiplier * overspeed^2 * dt
```

최고속도까지 빨리 가라고 reward를 주지는 않는다.

### Curve Speed Throttle Shaping

커브 감속은 reward만으로 기다리지 않고 control에도 직접 반영한다.
policy가 full throttle을 내더라도 전방 곡률 기반 허용속도를 넘으면 적용 throttle을 브레이크 쪽으로 깎는다.

```text
curve_speed_error = current_speed - v_allowed

if curve_speed_error > 0.1m/s:
    severity = clamp01((curve_speed_error - 0.1) / 1.5)
    curve_brake = lerp(0.0, -0.85, severity)
    applied_throttle = min(policy_throttle, curve_brake)
```

이 shaping은 코너 진입 전에 강제로 속도를 낮춰서, PPO가 코너 감속을 학습하기 전에도 물리적으로 커브를 못 도는 상황을 줄이는 목적이다.

### Roughness Penalty

action delta가 아니라 실제 차량 물리량을 본다.
현재는 비틀거림만 줄이기 위해 차체 기준 옆속도와 짧은 시간 안의 좌우 반전만 penalty로 쓴다. 횡가속, 종가속, 종방향 jerk penalty는 0이다.

```text
a_lat = local lateral acceleration
a_long = local longitudinal acceleration
j_long = local longitudinal jerk
v_lat = local lateral velocity
```

기본 margin:

```text
abs(a_lat)  <= 1.5m/s^2 -> no penalty
abs(a_long) <= 2.0m/s^2 -> no penalty
abs(j_long) <= 4.0m/s^3 -> no penalty
abs(v_lat)  <= 0.25m/s  -> no penalty
```

Penalty:

```text
roughness =
  0
    * max(0, abs(a_lat) - 1.5)^2
  + 0
    * max(0, abs(a_long) - 2.0)^2
  + 0
    * max(0, abs(j_long) - 4.0)^2
  + 0.12/sec
    * max(0, abs(v_lat) - 0.25)^2

reward -= roughness * dt
```

횡방향 흔들림은 lateral jerk 대신 차체 기준 옆속도와 짧은 시간 안의 좌우 반전으로 본다.
`v_lat` 부호가 threshold 이상에서 1초 안에 반대로 바뀌면 oscillation으로 보고 즉시 penalty를 준다.

```text
if sign(v_lat) flips
and abs(v_lat) > 0.25m/s
and time_since_previous_flip <= 1.0s:
    reward -= 0.03
```

가속도와 longitudinal jerk는 smoothing해서 사용한다.

```text
smoothed = lerp(previous, raw, 0.2)
```

## Steering Saturation Diagnostics

PPO action의 조향 target과 실제 적용 조향을 분리해서 기록한다.

```text
targetSteeringInput  = PPO action[0]
currentSteeringInput = smoothing 후 PlayerCar에 들어가는 값
```

기본 steering smoothing:

```text
steeringInputSmoothing = max(20, inspector value)
```

`FixedDeltaTime=0.02` 기준으로 0에서 full lock까지 약 0.06초가 걸린다.

PPO 제어 중에는 `PlayerCar.maxSteeringAngle`이 `maxSteeringAngleForFullInput`보다 작으면 런타임에 올린다.

```text
PlayerCar prefab default maxSteeringAngle = 30 deg
DRTPPOVehicleDriver maxSteeringAngleForFullInput = 45 deg

PPO control active:
  if PlayerCar.maxSteeringAngle < 45:
      PlayerCar.maxSteeringAngle = 45

PPO control released:
  restore original PlayerCar.maxSteeringAngle
```

TensorBoard stat:

```text
DRTDrive/TargetSteerAbs
DRTDrive/TargetSteer
DRTDrive/AppliedSteerAbs
DRTDrive/SteerLagAbs
DRTDrive/AppliedSteerDeg
DRTDrive/MaxSteerDeg
DRTDrive/TargetSteerSaturated
DRTDrive/AppliedSteerSaturated
DRTDrive/SteeringCapacityOverride
DRTDrive/TargetThrottle
DRTDrive/AppliedThrottle
DRTDrive/SpeedMS
DRTDrive/ForwardSpeedMS
DRTDrive/SignedLateralError
DRTDrive/HeadingCross
DRTDrive/CurveAllowedSpeedMS
DRTDrive/CurveSpeedErrorMS
DRTDrive/CurveSpeedBrakeApplied
DRTDrive/AllowedSpeedMS
DRTDrive/OverspeedPenalty
DRTDrive/CurveOverspeedPenalty
DRTDrive/FrontCurveStrength
DRTDrive/LocalLateralVelocityAbsMS
DRTDrive/LocalLateralVelocityPenalty
DRTDrive/LateralOscillationPenalty
DRTDrive/LateralOscillationFlip
DRTDrive/IdleStopped
DRTDrive/IdleLowSpeed
DRTDrive/IdleProgressSpeed
DRTDrive/IdleProgressWindowDistance
DRTDrive/IdleNearDestination
DRTDrive/IdleFrontBlocked
DRTDrive/IdleFrontBlockClearance
DRTDrive/UnblockedIdleSeconds
DRTDrive/UnblockedIdlePenalty
DRTDrive/TrackingLookaheadMeters
DRTDrive/LookaheadBearingDeg
DRTDrive/GeometricSteerDemandAbs
DRTDrive/GeometricSteerDemandSaturated
```

해석:

```text
TargetSteerSaturated 높고 AppliedSteerSaturated 낮음:
  smoothing lag 때문에 실제 조향이 target을 못 따라감

TargetSteerSaturated 높고 AppliedSteerSaturated 높고 ADE도 큼:
  실제 max steer, 속도, 차량 물리 한계 문제 가능성

GeometricSteerDemandSaturated 높음:
  reference path/lookahead 기준으로도 full steer에 가까운 조향이 필요함

TargetSteerSaturated 낮고 커브를 못 돎:
  policy가 충분한 steer action을 아직 못 배웠거나 reward 균형 문제 가능성
```

## Episode Termination

### 목적지 도착

```text
reward += +2
EndEpisode()
```

### 충돌

```text
reward += -2
EndEpisode()
```

### Reference Path 과이탈

```text
if e_y > 4.0m:
    reward += -1.5
    EndEpisode()
```

### 30초 무이동

```text
if vehicle position does not change meaningfully for 30s:
    EndEpisode()
```

무이동 종료에는 추가 penalty를 주지 않는다.

## 사용하지 않는 항목

현재 설계에서 제외한 항목:

```text
generic stop penalty
stuck penalty
road boundary termination
Gley traffic network exit termination for PPO training
ray proximity penalty
vehicle clearance reward penalty, default coefficient 0
unblocked idle reward penalty, default coefficient 0
overspeed reward penalty, default coefficient 0
lateral/longitudinal acceleration and longitudinal jerk roughness penalty
speed tracking reward
action delta smoothing penalty
```
