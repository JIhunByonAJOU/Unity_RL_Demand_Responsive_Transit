# DRT PPO PurePursuit State / Reward / Penalty

## Unity Inspector 파라미터

아래 항목은 `DRTBusController`의 `PPO PurePursuit Parameters` 섹션에서 조정하는 값이다. 이 값들은 실행 시 `DRTPPOPurePursuitVehicleDriver`로 전달되어 PPO PurePursuit 학습/추론 주행에 적용된다.

| Inspector 이름 | 코드 변수 | 기본값 | 설명 |
| --- | --- | ---: | --- |
| `Min Ld (m)` | `ppoPurePursuitMinLookaheadMeters` -> `minLookaheadMeters` | 0.1 | PPO가 선택할 수 있는 최소 lookahead distance. 너무 작으면 조향 명령이 민감해지고 오실레이션이 커질 수 있다. |
| `Max Ld (m)` | `ppoPurePursuitMaxLookaheadMeters` -> `maxLookaheadMeters` | 25 | PPO가 선택할 수 있는 최대 lookahead distance. 고속 직진 안정성에 유리하지만 코너 추종은 둔해질 수 있다. |
| `Zero Action Ld Norm` | `ppoPurePursuitZeroActionLookaheadNormalized` -> `zeroActionLookaheadNormalized` | 0 | ML-Agents continuous action `0`이 들어왔을 때의 Ld 정규화 위치. `0`이면 최소 Ld, `0.05`면 전체 범위의 5% 지점에서 시작한다. |
| `Min Speed Ratio` | `ppoPurePursuitMinTargetSpeedRatio` -> `minTargetSpeedRatio` | 0 | PPO가 선택할 수 있는 최소 목표 속도 비율. 최종 목표 속도는 `roadSpeed * speedRatio`로 계산된다. |
| `Max Speed Ratio` | `ppoPurePursuitMaxTargetSpeedRatio` -> `maxTargetSpeedRatio` | 1 | PPO가 선택할 수 있는 최대 목표 속도 비율. 기본값 `1`은 Gley 도로 제한속도까지 선택 가능하다는 뜻이다. |
| `Zero Action Speed Norm` | `ppoPurePursuitZeroActionSpeedNormalized` -> `zeroActionSpeedNormalized` | 1 | ML-Agents continuous action `0`이 들어왔을 때의 속도 비율 정규화 위치. `1`이면 초기 action 0에서 도로 제한속도를 목표로 한다. |
| `Throttle Smooth` | `ppoPurePursuitThrottleInputSmoothing` -> `throttleInputSmoothing` | 6 | 목표 속도 추종을 위해 throttle/brake 입력이 변하는 최대 속도. 높을수록 target throttle/brake에 빠르게 붙는다. |
| `Steering Smooth` | `ppoPurePursuitSteeringInputSmoothing` -> `steeringInputSmoothing` | 12 | Pure Pursuit가 만든 목표 조향 입력으로 실제 external steering input이 따라가는 속도. 낮으면 조향 반응이 둔하고, 높으면 급격해질 수 있다. |
| `Curvature Smooth Beta` | `ppoPurePursuitCurvatureSmoothingBeta` -> `curvatureSmoothingBeta` | 0.75 | Pure Pursuit curvature command smoothing 계수. `1`에 가까울수록 새 curvature 명령을 거의 그대로 반영한다. |
| `Speed Reward` | `ppoPurePursuitSpeedRewardPerSecond` -> `speedRewardPerSecond` | 0.005 | 현재 속도에 비례한 약한 보상. 차가 서 있는 정책을 줄이되, 코너에서 무조건 빠르게 달리는 쪽으로 치우치지 않도록 낮게 둔다. |
| `Progress Reward` | `ppoPurePursuitProgressRewardPerMeter` -> `progressRewardPerMeter` | 0.1 | Frenet `s` 기준으로 reference path를 따라 앞으로 진행한 거리 보상. 목적지 직선거리보다 이 항목이 주된 전진 보상이다. |
| `Destination Progress Reward` | `ppoPurePursuitDestinationProgressRewardPerMeter` -> `destinationProgressRewardPerMeter` | 0 | 목적지 직선거리 감소 보상. 코너 안쪽을 깎아먹는 행동을 보상할 수 있어서 기본값은 0으로 비활성화한다. |
| `Destination Reward` | `ppoPurePursuitDestinationReward` -> `destinationReward` | 10 | 목적지 도착 시 한 번 주는 terminal reward. 중간 직선거리 보상 대신 정상 도착을 크게 보상한다. |
| `Ld Change Penalty` | `ppoPurePursuitLookaheadChangePenaltyPerMeter` -> `lookaheadChangePenaltyPerMeter` | 0 | 연속 timestep 사이의 Ld 변화량 패널티. 현재는 Ld를 곡률/속도에 따라 자유롭게 바꾸도록 기본값 0이다. |
| `Lateral Penalty / m` | `ppoPurePursuitLateralErrorPenaltyPerMeter` -> `lateralErrorPenaltyPerMeter` | 1.5 | Frenet 횡방향 오차에 비례한 거리 기준 패널티 가중치. 경로 안쪽을 깎거나 바깥으로 밀리는 행동을 줄인다. |
| `Local Lat Vel Penalty / m` | `ppoPurePursuitLocalLateralVelocityPenaltyPerMeter` -> `localLateralVelocityPenaltyPerMeter` | 1 | 차량 local X축 횡속도 절댓값에 비례한 거리 기준 패널티. 차체가 옆으로 흐르거나 조향 오실레이션이 커지는 거동을 줄인다. |
| `Local Lat Vel Kappa Gain` | `ppoPurePursuitLocalLateralVelocityCurvatureGain` -> `localLateralVelocityCurvatureGain` | 3 | local lateral velocity 패널티에 곡률을 곱해 키우는 계수. 큰 곡률 구간에서 횡속도/슬립이 생기면 더 크게 벌점이 들어간다. |
| `Heading Penalty / m` | `ppoPurePursuitHeadingErrorPenaltyPerMeter` -> `headingErrorPenaltyPerMeter` | 0.05 | 차량 heading과 path tangent 사이의 오차에 비례한 거리 기준 패널티 가중치. 진행 방향이 경로 방향과 크게 어긋나는 상태를 줄인다. |
| `Overspeed Penalty` | `ppoPurePursuitOverspeedPenaltyPerSecond` -> `overspeedPenaltyPerSecond` | 0.3 | 실제 속도가 Gley 도로 제한속도를 넘었을 때의 패널티 가중치. 제한속도 이하에서 코너가 너무 빠른 경우는 이 항목만으로는 벌하지 않는다. |
| `CrossTrack Norm (m)` | `ppoPurePursuitMaxCrossTrackErrorMeters` -> `maxCrossTrackErrorMeters` | 6 | 횡방향 오차 observation 및 lateral penalty 정규화 기준 거리. 작게 잡으면 같은 오차도 더 크게 정규화된다. |
| `CrossTrack End (m)` | `ppoPurePursuitHardCrossTrackLimitMeters` -> `hardCrossTrackLimitMeters` | 4 | Frenet 횡방향 오차가 이 값 이상이면 episode를 종료한다. |
| `CrossTrack End Penalty` | `ppoPurePursuitAssignedRouteExitPenaltyMagnitude` -> `assignedRouteExitPenalty` | 5 | 횡방향 오차 종료 시 주는 terminal penalty의 크기. Inspector에는 양수 `5`로 넣고, 내부 reward에는 `-5`로 적용한다. |
| `No Move Timeout (s)` | `ppoPurePursuitNoMovementTimeoutRealSeconds` -> `noMovementTimeoutRealSeconds` | 30 | 지정 시간 동안 충분히 움직이지 못하면 episode를 종료한다. |
| `Min Move (m)` | `ppoPurePursuitMinimumMovementMeters` -> `minimumMovementMeters` | 0.25 | no-movement 판단에서 "움직였다"고 인정할 최소 이동 거리. |

### 횡방향 이탈 종료 관련 이름

횡방향 오차가 너무 커져서 episode가 종료될 때는 threshold와 penalty가 분리되어 있다.

```text
종료 기준 Inspector 이름: CrossTrack End (m)
종료 기준 코드 변수: hardCrossTrackLimitMeters
기본값: 4m

종료 벌점 코드 변수: assignedRouteExitPenalty
Inspector 이름: CrossTrack End Penalty
Inspector 기본값: 5
내부 적용값: -5
```

## 목적

`DRTPPOPurePursuitVehicleDriver`는 기존 `DRTPPOVehicleDriver`를 건드리지 않고 별도 Physical Drive 모드인 `PPO PurePursuit`에서만 사용한다.

현재 버전은 논문식 `Ld-only` 구조를 DRT 도로 주행에 맞게 확장한 형태다.

- PPO가 Pure Pursuit의 lookahead distance `Ld`를 직접 선택한다.
- PPO가 도로 제한속도 대비 목표 속도 비율도 직접 선택한다.
- 조향각은 Pure Pursuit 기하식으로 계산한다.
- throttle/brake 입력은 PPO가 직접 내지 않고, PPO가 고른 목표 속도를 low-level speed controller가 추종한다.

따라서 핵심 학습 목표는 다음과 같다.

```text
전방 곡률, 현재 속도, 횡방향 오차, heading 오차를 보고
적절한 Ld와 목표 속도 비율을 함께 선택해서
도로 제한속도 안에서 빠르게 진행하되 Frenet 횡방향 오차를 줄인다.
```

## Unity 모드

Inspector에서 `DRTBusController`의 Physical Drive Mode를 다음 값으로 선택한다.

```text
PPO PurePursuit
```

실행 시 player vehicle 아래에 별도 agent object가 생성된다.

```text
DRTPurePursuitPPOAgent
```

Behavior name은 다음과 같다.

```text
DRTPurePursuitPPO
```

## Action

PPO action은 연속값 2개다.

```text
action[0] = lookahead command in [-1, 1]
action[1] = target speed ratio command in [-1, 1]
```

### Lookahead Action

```text
if action[0] <= 0:
  u_ld = lerp(0, zeroActionLookaheadNormalized, action[0] + 1)
else:
  u_ld = lerp(zeroActionLookaheadNormalized, 1, action[0])

Ld_cmd = lerp(minLookaheadMeters, maxLookaheadMeters, u_ld)
Ld     = clamp(Ld_cmd, minLookaheadMeters, maxLookaheadMeters)
```

기본값:

```text
minLookaheadMeters = 0.1m
maxLookaheadMeters = 25m
```

`Ld` smoothing은 사용하지 않는다. 현재 tick의 action이 바로 적용 `Ld`가 된다.

초기 PPO policy가 보통 `action[0] ~= 0`을 내므로, `zeroActionLookaheadNormalized`가 학습 극초기 `Ld`를 결정한다.

```text
zeroActionLookaheadNormalized = 0     -> action[0] = 0에서 최소 Ld
zeroActionLookaheadNormalized = 0.125 -> action[0] = 0에서 약 3.2m
action[0] = 1                         -> 최대 Ld
```

Pure Pursuit 조향 계산:

```text
curvature = 2 * local_target_x / (Ld * Ld)
steer_deg = atan(wheel_base * curvature)
steer     = clamp(steer_deg / max_steer_deg, -1, 1)
```

단, Pure Pursuit 곡률 명령과 steering input에는 차량 입력 안정화를 위한 smoothing이 남아 있다.

### Speed Action

```text
if action[1] <= 0:
  u_speed = lerp(0, zeroActionSpeedNormalized, action[1] + 1)
else:
  u_speed = lerp(zeroActionSpeedNormalized, 1, action[1])

speed_ratio = lerp(minTargetSpeedRatio, maxTargetSpeedRatio, u_speed)
target_speed = road_speed * speed_ratio
```

기본값:

```text
minTargetSpeedRatio = 0.0
maxTargetSpeedRatio = 1.0
```

`road_speed`는 Gley waypoint의 `MaxSpeed`를 m/s로 변환한 값이다. Gley 속도값이 없을 때만 fallback speed를 쓴다.

초기 PPO policy가 보통 `action[1] ~= 0`을 내므로, `zeroActionSpeedNormalized`가 학습 극초기 목표 속도 비율을 결정한다.

```text
zeroActionSpeedNormalized = 1   -> action[1] = 0에서 도로 제한속도
zeroActionSpeedNormalized = 0.6 -> action[1] = 0에서 제한속도 60%
action[1] = -1                 -> 최소 speed ratio
action[1] = 1                  -> 최대 speed ratio
```

```text
road_speed = Gley waypoint MaxSpeed / 3.6

if Gley MaxSpeed is missing:
  road_speed = fallback speed
```

현재 버전에서는 곡률 기반 `curve_speed = sqrt(a_lat / kappa)`로 목표 속도를 강제로 낮추지 않는다. 전방 곡률은 state와 reward shaping에 들어가며, PPO가 곡률을 보고 `Ld`와 `speed_ratio`를 함께 고르게 한다.

목적지 접근 구간에서는 안전 정지를 위해 기존 destination slowdown은 유지한다.

## Reference Path

Gley route를 그대로 추종하지 않고, route waypoint와 목적지를 이용해 1m 간격 reference path를 만든다.

```text
raw route:
  current vehicle position
  Gley waypoint positions
  destination
  destination tail point

reference path:
  raw route를 referenceWaypointSpacingMeters 간격으로 resampling
```

기본값:

```text
referenceWaypointSpacingMeters      = 1.0m
referenceWaypointPassDistanceMeters = 0.5m
```

차량 위치는 매 tick reference path에 Frenet projection된다.

```text
s   = reference path 진행 거리
e_y = reference path 기준 횡방향 오차
```

## State

Vector observation size는 8이다.

```text
1. speed_norm
2. kappa_near_norm
3. kappa_mid_norm
4. kappa_far_norm
5. delta_kappa_norm
6. lateral_error_norm
7. heading_error_norm
8. road_speed_norm
```

정규화:

```text
speed_norm         = current_speed / maxObservationSpeedMetersPerSecond
kappa_near_norm    = max_curvature(s + 0m  .. s + 6m)  / maxObservationCurvature
kappa_mid_norm     = max_curvature(s + 6m  .. s + 12m) / maxObservationCurvature
kappa_far_norm     = max_curvature(s + 12m .. s + 24m) / maxObservationCurvature
delta_kappa_norm   = (kappa_mid - kappa_near) / maxObservationCurvature
lateral_error_norm = abs(e_y) / maxCrossTrackErrorMeters
heading_error_norm = heading_error_deg / 180
road_speed_norm    = road_speed / maxObservationSpeedMetersPerSecond
```

곡률 기본값:

```text
midCurvatureHorizonMeters = 6m
farCurvatureHorizonMeters = 12m
maxCurvatureHorizonMeters = 24m
curvatureSampleSpacingMeters = 1m
```

Gley waypoint는 smooth raceline이 아니라 polyline에 가깝기 때문에 단일 지점 curvature를 쓰지 않는다. 각 horizon 구간 안에서 1m 간격으로 curvature를 스캔하고, 그 구간의 max curvature를 state에 넣는다.

## Reward / Penalty

전체 보상은 다음 구조다.

```text
R_t =
  + speed_reward
  + path_progress_reward
  + waypoint_pass_reward
  + destination_progress_reward
  - lookahead_teacher_penalty
  - lookahead_change_penalty
  - curvature_penalty
  - lateral_error_penalty
  - local_lateral_velocity_penalty
  - heading_error_penalty
  - overspeed_penalty
  - stall_penalty
```

### speed_reward

```text
speed_reward = speedRewardPerSecond * current_speed * dt
```

차가 아예 안 가는 정책으로 수렴하지 않게 하는 약한 shaping이다. 코너에서 무조건 빠르게 달리는 정책을 막기 위해 기본 가중치는 낮게 둔다.

### path_progress_reward

```text
path_progress_reward =
  progressRewardPerMeter * max(0, current_s - previous_s)
```

Frenet `s` 기준으로 reference path를 따라 앞으로 진행했을 때만 보상한다.

### waypoint_pass_reward

```text
waypoint_pass_reward =
  waypointPassedReward * passed_reference_point_count
```

1m resampled reference point를 순차 통과하도록 돕는다.

### destination_progress_reward

```text
destination_progress_reward =
  destinationProgressRewardPerMeter * max(0, previous_destination_distance - current_destination_distance)
```

현재 기본값은 `0`이다. 목적지 직선거리 감소 보상은 코너 안쪽을 깎아먹는 행동을 보상할 수 있으므로 기본 학습에서는 비활성화한다. 최종 목적지에 도착했을 때만 terminal reward를 준다.

### lookahead_teacher_penalty

```text
Ld_teacher =
  clamp(
    teacherBaseLookaheadMeters
    + teacherSpeedGainSeconds * current_speed
    - teacherCurvatureGain * maxKappa,
    minLookaheadMeters,
    maxLookaheadMeters
  )

lookahead_teacher_penalty =
  lookaheadTeacherPenaltyPerMeter * abs(Ld - Ld_teacher) * dt
```

기본값은 `0`으로 꺼져 있다. policy가 직접 `Ld`를 고르는 모습을 보기 위해 teacher shaping은 기본 비활성화 상태다.

### lookahead_change_penalty

```text
lookahead_change_penalty =
  lookaheadChangePenaltyPerMeter * abs(Ld_t - Ld_t-1)
```

현재 기본값은 `0`이다. 이 프로젝트의 핵심은 곡률과 속도에 따라 `Ld`를 자유롭게 바꾸는 것이므로, 학습 초반에는 `Ld` 변화 자체를 벌하지 않는다. 조향 튐은 Pure Pursuit curvature smoothing, steering input smoothing, 차량 물리로 완화한다.

### curvature_penalty

```text
curvature_penalty =
  curvaturePenaltyPerSecond * maxKappa * dt
```

곡률이 큰 구간에서 무리하게 빠른 주행만 추구하는 정책을 약하게 억제하는 shaping이다. 실제 목표 속도는 PPO의 `speed_ratio`가 결정한다.

### lateral_error_penalty

```text
lateral_error_norm =
  clamp01(abs(e_y) / maxCrossTrackErrorMeters)

lateral_error_penalty =
  lateralErrorPenaltyPerMeter
  * lateral_error_norm^2
  * path_progress_meters
```

도로 boundary 대신 reference path의 Frenet 횡방향 오차를 본다. 코너 안쪽을 깎아먹는 정책을 줄이기 위한 핵심 패널티다. 시간 기준이 아니라 Frenet 진행거리 기준으로 누적되므로, 빠르게 통과해서 패널티 시간을 줄이는 행동을 덜 유도한다.

### local_lateral_velocity_penalty

```text
local_lateral_velocity =
  abs(vehicle_root.InverseTransformDirection(rigidbody_velocity).x)

curvature_multiplier =
  1 + localLateralVelocityCurvatureGain * maxKappa

local_lateral_velocity_penalty =
  localLateralVelocityPenaltyPerMeter
  * local_lateral_velocity
  * curvature_multiplier
  * path_progress_meters
```

차량 local 좌표계의 X축 횡속도를 본다. 값이 크면 차량이 진행 방향으로만 움직이는 것이 아니라 옆으로 흐르거나 좌우 오실레이션이 커진 상태로 본다. 큰 곡률 구간에서는 `maxKappa` multiplier 때문에 같은 횡속도라도 더 큰 패널티를 받는다. 시간 기준이 아니라 Frenet 진행거리 기준으로 누적되므로, 코너를 빨리 통과해서 패널티 시간을 줄이는 행동을 덜 유도한다.

### heading_error_penalty

```text
heading_error_norm =
  clamp01(abs(heading_error_deg) / 180)

heading_error_penalty =
  headingErrorPenaltyPerMeter
  * heading_error_norm
  * path_progress_meters
```

차량 heading이 path tangent와 크게 어긋난 상태를 줄인다. 시간 기준이 아니라 Frenet 진행거리 기준으로 누적한다.

### overspeed_penalty

```text
overspeed =
  max(0, current_speed - road_speed)

overspeed_penalty =
  overspeedPenaltyPerSecond * overspeed^2 * dt
```

도로 제한속도 초과만 벌한다. 곡률 기반 허용속도는 더 이상 hard cap으로 쓰지 않는다.

### stall_penalty

```text
if current_speed < stallSpeedThresholdMetersPerSecond
   for longer than stallGraceSeconds:
     stall_penalty = stallPenaltyPerSecond * dt
```

정지 상태로 버티는 정책을 줄인다.

## Episode 종료 조건

목적지 도착:

```text
+ destinationReward
EndEpisode()
```

충돌:

```text
+ collisionPenalty
EndEpisode()
```

Frenet 횡방향 오차 4m 이상:

```text
if lateral_error >= hardCrossTrackLimitMeters:
  + assignedRouteExitPenalty
  EndEpisode()
```

reference path fault:

```text
+ referenceFaultPenalty
EndEpisode()
```

30초 이상 같은 자리에서 못 움직임:

```text
noMovementTimeoutRealSeconds = 30s
minimumMovementMeters        = 0.25m
```

차가 브레이크등만 깜빡이거나 움찔거리더라도 30초 동안 누적 위치가 0.25m 이상 갱신되지 않으면 에피소드가 끝난다. 이 종료 조건에는 별도 terminal penalty를 추가하지 않는다.

## Inspector 파라미터

학습 시작 전에 조정할 값은 `DRTBusController` 컴포넌트의 `PPO PurePursuit Parameters` 섹션에 있다. `DRTPurePursuitPPOAgent`는 Play Mode에서 런타임 생성되므로, 직접 child agent를 찾아 값을 넣는 방식이 아니라 BusController에 값을 넣고 시작한다.

런타임에 BusController가 생성된 `DRTPPOPurePursuitVehicleDriver`로 이 값을 복사한다.

### Lookahead

| 변수 | 기본값 | 의미 |
| --- | ---: | --- |
| `minLookaheadMeters` | 0.1 | PPO가 선택할 수 있는 최소 `Ld` |
| `maxLookaheadMeters` | 25 | PPO가 선택할 수 있는 최대 `Ld` |
| `zeroActionLookaheadNormalized` | 0 | `action[0] = 0`일 때의 `Ld` 정규화 위치. 0이면 최소 `Ld`, 0.125면 약 3.2m |
| `teacherBaseLookaheadMeters` | 4 | teacher `Ld` 기본값 |
| `teacherSpeedGainSeconds` | 0.5 | 속도 증가에 따른 teacher `Ld` 증가 |
| `teacherCurvatureGain` | 10 | 곡률 증가에 따른 teacher `Ld` 감소 |

### Pure Pursuit Control

| 변수 | 기본값 | 의미 |
| --- | ---: | --- |
| `curvatureSmoothingBeta` | 0.75 | Pure Pursuit 곡률 명령 smoothing 계수 |
| `steeringInputSmoothing` | 12 | steering input smoothing 속도 |
| `throttleInputSmoothing` | 6 | throttle/brake input smoothing 속도 |

### Speed Control

| 변수 | 기본값 | 의미 |
| --- | ---: | --- |
| `baseCruiseSpeedMetersPerSecond` | 5 | Gley `MaxSpeed`가 없을 때만 쓰는 fallback road speed |
| `usePolicySpeedLimit` | true | Gley `MaxSpeed`가 없을 때 fallback PPO speed 사용 여부 |
| `maxPolicySpeedMetersPerSecond` | 5 | Gley `MaxSpeed`가 없을 때 쓰는 fallback PPO speed |
| `minTargetSpeedRatio` | 0 | PPO가 선택할 수 있는 최소 도로 제한속도 비율 |
| `maxTargetSpeedRatio` | 1 | PPO가 선택할 수 있는 최대 도로 제한속도 비율 |
| `zeroActionSpeedNormalized` | 1 | `action[1] = 0`일 때의 speed ratio 정규화 위치. 1이면 도로 제한속도, 0.6이면 제한속도 60% |
| `slowDownDistanceMeters` | 10 | 목적지 접근 감속 거리 |

### Reward Weights

| 변수 | 기본값 | 의미 |
| --- | ---: | --- |
| `speedRewardPerSecond` | 0.005 | 속도 보상 |
| `progressRewardPerMeter` | 0.1 | Frenet `s` 진행 보상 |
| `waypointPassedReward` | 0.05 | 1m reference point 통과 보상 |
| `destinationProgressRewardPerMeter` | 0 | 목적지 접근 보상. 기본 비활성 |
| `lookaheadTeacherPenaltyPerMeter` | 0 | teacher `Ld`와의 차이 패널티 |
| `lookaheadChangePenaltyPerMeter` | 0 | `Ld` 변화량 패널티 |
| `curvaturePenaltyPerSecond` | 0.02 | 곡률 구간 shaping 패널티 |
| `lateralErrorPenaltyPerMeter` | 1.5 | Frenet 횡방향 오차 거리 기준 패널티 |
| `localLateralVelocityPenaltyPerMeter` | 1 | 차량 local 횡속도 거리 기준 패널티 |
| `localLateralVelocityCurvatureGain` | 3 | local 횡속도 패널티의 곡률 multiplier |
| `headingErrorPenaltyPerMeter` | 0.05 | path tangent 대비 heading 오차 거리 기준 패널티 |
| `overspeedPenaltyPerSecond` | 0.3 | 도로 제한속도 초과 패널티 |
| `stallPenaltyPerSecond` | 0.05 | 정지 상태 지속 패널티 |
| `destinationReward` | 10 | 목적지 도착 terminal reward |
| `collisionPenalty` | -2 | 충돌 terminal penalty |
| `assignedRouteExitPenalty` | -5 | 횡방향 4m 이탈 terminal penalty |
| `referenceFaultPenalty` | -1 | reference fault terminal penalty |

### Termination

| 변수 | 기본값 | 의미 |
| --- | ---: | --- |
| `hardCrossTrackLimitMeters` | 4 | Frenet 횡방향 오차 종료 기준 |
| `noMovementTimeoutRealSeconds` | 30 | 한 자리 no movement 종료 시간 |
| `minimumMovementMeters` | 0.25 | 움직임 갱신으로 인정할 최소 이동거리 |
| `stallSpeedThresholdMetersPerSecond` | 0.05 | stall penalty 속도 기준 |
| `stallGraceSeconds` | 2 | stall penalty 유예 시간 |

## 현재 구현 요약

```text
Action:
  2 continuous actions
  action[0] -> Ld increase command. 0 starts at min Ld.
  action[1] -> speed reduction command. 0 starts at road speed.

State:
  speed
  near/mid/far curvature
  delta curvature
  lateral error
  heading error
  road speed

Speed:
  road_speed = Gley MaxSpeed
  target_speed = road_speed * PPO speed_ratio
  curve_speed hard cap 없음

Safety:
  current_speed > road_speed 이면 overspeed penalty
  lateral_error >= 4m 이면 episode 종료
```
