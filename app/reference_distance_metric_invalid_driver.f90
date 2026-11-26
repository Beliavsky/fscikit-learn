program reference_distance_metric_invalid_driver
 use sklearn_common,only:dp,require
 use sklearn_metrics,only:distance_metric,get_metric
 use,intrinsic::ieee_arithmetic
 implicit none
 type(distance_metric)::metric
 real(dp)::x(2,2),y(1,3),bad,inf
 real(dp),allocatable::d(:,:)
 character(64)::name
 call get_command_argument(1,name)
 x=1;y=1;bad=ieee_value(0._dp,ieee_quiet_nan);inf=ieee_value(0._dp,ieee_positive_inf)
 select case(trim(name))
 case('unknown');metric=get_metric('cosine')
 case('unused_p');metric=get_metric('euclidean',p=2._dp)
 case('mink_zero');metric=get_metric('minkowski',p=0._dp)
 case('mink_negative');metric=get_metric('minkowski',p=-1._dp)
 case('weighted_inf');metric=get_metric('minkowski',p=inf,w=[1._dp,1._dp])
 case('weights_empty');metric=get_metric('minkowski',p=2._dp,w=[real(dp)::])
 case('weights_negative');metric=get_metric('minkowski',w=[-1._dp,1._dp])
 case('weights_nan');metric=get_metric('minkowski',w=[bad,1._dp])
 case('weights_inf');metric=get_metric('minkowski',w=[inf,1._dp])
 case('weights_features')
  metric=get_metric('minkowski',w=[1._dp]);d=metric%pairwise(x)
 case('variance_missing');metric=get_metric('seuclidean')
 case('variance_rank');metric=get_metric('seuclidean',v=x)
 case('variance_features')
  metric=get_metric('seuclidean',v=[1._dp]);d=metric%pairwise(x)
 case('precision_missing');metric=get_metric('mahalanobis')
 case('precision_square');metric=get_metric('mahalanobis',vi=y)
 case('precision_features')
  metric=get_metric('mahalanobis',vi=reshape([1._dp],[1,1]));d=metric%pairwise(x)
 case('covariance_singular');metric=get_metric('mahalanobis',v=x)
 case('haversine_features')
  metric=get_metric('haversine');d=metric%pairwise(y)
 case('input_nan')
  metric=get_metric('euclidean');x(1,1)=bad;d=metric%pairwise(x)
 case('input_inf')
  metric=get_metric('euclidean');x(1,1)=inf;d=metric%pairwise(x)
 case('input_empty')
  metric=get_metric('euclidean');d=metric%pairwise(x(:0,:))
 case('input_features')
  metric=get_metric('euclidean');d=metric%pairwise(x,y)
 case('callback_minus_one')
  metric=get_metric(callback);d=metric%pairwise(x)
 case default;call require(.false.,'unknown invalid DistanceMetric case')
 end select
 call require(.false.,'invalid DistanceMetric input was accepted')
contains
 function callback(u,v)result(value)
  real(dp),intent(in)::u(:),v(:)
  real(dp)::value
  value=-1
 end function
end program
