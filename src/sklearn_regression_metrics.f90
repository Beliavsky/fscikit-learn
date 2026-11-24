! sklearn 1.6.1 metrics/_regression.py and utils/stats.py contracts (BSD).
! Native rank adapters and quiet IEEE composition are original MIT contributions.
module sklearn_regression_metrics
  use sklearn_common, only: dp,require
  use sklearn_ieee_math, only: ieee_difference,ieee_product,ieee_quotient,ieee_total,ieee_mean
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value, &
    ieee_quiet_nan,ieee_positive_inf,ieee_negative_inf
  implicit none
  private
  public :: mean_pinball_loss,mean_tweedie_deviance,d2_tweedie_score,d2_pinball_score,d2_absolute_error_score
  public :: root_mean_squared_log_error
  interface mean_pinball_loss
    module procedure mean_pinball_loss_vector,mean_pinball_loss_matrix
  end interface
  interface d2_pinball_score
    module procedure d2_pinball_score_vector,d2_pinball_score_matrix
  end interface
  interface d2_absolute_error_score
    module procedure d2_absolute_error_score_vector,d2_absolute_error_score_matrix
  end interface
  interface root_mean_squared_log_error
    module procedure root_mean_squared_log_error_vector,root_mean_squared_log_error_matrix
  end interface
contains
  function ascending_order(scores) result(order)
    real(dp), intent(in) :: scores(:)
    integer :: order(size(scores)),work(size(scores)),n,width,left,middle,right,i,j,k
    n=size(scores);order=[(i,i=1,n)];width=1
    do while(width<n)
      left=1
      do while(left<=n)
        middle=left+min(width,n-left+1);right=middle+min(width,n-middle+1)
        i=left;j=middle
        do k=left,right-1
          if(i>=middle) then
            work(k)=order(j);j=j+1
          else if(j>=right) then
            work(k)=order(i);i=i+1
          else if(scores(order(i))<=scores(order(j))) then
            work(k)=order(i);i=i+1
          else
            work(k)=order(j);j=j+1
          end if
        end do
        left=right
      end do
      order=work
      if(width>=n-width) exit
      width=width*2
    end do
  end function
  real(dp) function nan_value() result(value)
    value=ieee_value(0._dp,ieee_quiet_nan)
  end function
  subroutine check_arrays(y,predicted,w,check_mass)
    real(dp), intent(in) :: y(:,:),predicted(:,:)
    real(dp), optional, intent(in) :: w(:)
    logical, optional, intent(in) :: check_mass
    logical :: mass
    mass=.true.;if(present(check_mass)) mass=check_mass
    call require(size(y,1)>0.and.size(y,2)>0,'regression metrics require nonempty targets')
    call require(all(shape(y)==shape(predicted)),'regression metric target/prediction shapes differ')
    call require(all(ieee_is_finite(y)).and.all(ieee_is_finite(predicted)), &
      'regression metric targets and predictions must be finite')
    if(present(w)) then
      call require(size(w)==size(y,1),'regression metric sample weight length mismatch')
      call require(all(ieee_is_finite(w)),'regression metric sample weights must be finite')
      if(mass) call require(ieee_total(w)/=0,'regression metric sample weights must have nonzero total')
    end if
  end subroutine
  subroutine check_alpha(alpha)
    real(dp), intent(in) :: alpha
    call require(ieee_is_finite(alpha),'pinball alpha must be finite')
    call require(alpha>=0.and.alpha<=1,'pinball alpha must be in [0,1]')
  end subroutine
  real(dp) function aggregate(values,output_weights,raw_values) result(value)
    real(dp), intent(in) :: values(:)
    real(dp), optional, intent(in) :: output_weights(:)
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    if(present(output_weights)) then
      call require(size(output_weights)==size(values),'regression metric output weight length mismatch')
      call require(all(ieee_is_finite(output_weights)),'regression metric output weights must be finite')
      call require(ieee_total(output_weights)/=0,'regression metric output weights must have nonzero total')
    end if
    if(present(raw_values)) raw_values=values
    value=ieee_mean(values,output_weights)
  end function
  real(dp) function pinball(y,predicted,alpha,w) result(value)
    real(dp), intent(in) :: y(:),predicted(:),alpha
    real(dp), optional, intent(in) :: w(:)
    real(dp) :: losses(size(y)),difference,sign_
    integer :: i
    do i=1,size(y)
      difference=ieee_difference(y(i),predicted(i));sign_=0
      if(.not.ieee_is_nan(difference)) then
        if(difference>=0) sign_=1
      end if
      losses(i)=ieee_difference(ieee_product(alpha*sign_,difference), &
        ieee_product((1-alpha)*(1-sign_),difference))
    end do
    value=ieee_mean(losses,w)
  end function
  real(dp) function quantile(y,alpha,w) result(value)
    real(dp), intent(in) :: y(:),alpha
    real(dp), optional, intent(in) :: w(:)
    integer :: order(size(y)),n,left,right,middle,index
    real(dp) :: cumulative(size(y)),position,fraction,difference,target
    n=size(y);order=ascending_order(y)
    if(present(w)) then
      cumulative(1)=w(order(1))
      do index=2,n
        cumulative(index)=ieee_difference(cumulative(index-1),-w(order(index)))
      end do
      ! sklearn computes percentile as alpha*100, then divides by 100.
      target=ieee_product((alpha*100)/100,cumulative(n))
      if(target==0) target=nearest(0._dp,1._dp)
      ! Match searchsorted's binary search even for signed-weight CDFs.
      left=1;right=n+1
      do while(left<right)
        middle=left+(right-left)/2
        if(middle<=n) then
          if(ieee_is_nan(target)) then
            if(ieee_is_nan(cumulative(middle))) then
              right=middle
            else
              left=middle+1
            end if
          else if(ieee_is_nan(cumulative(middle))) then
            right=middle
          else if(cumulative(middle)<target) then
            left=middle+1
          else
            right=middle
          end if
        else
          right=middle
        end if
      end do
      value=y(order(min(left,n)))
    else
      position=(alpha*100)/100*(n-1);index=floor(position)+1;fraction=position-floor(position)
      if(index>=n) then
        value=y(order(n));return
      end if
      difference=ieee_difference(y(order(index+1)),y(order(index)))
      value=ieee_difference(y(order(index)),-ieee_product(difference,fraction))
      if(fraction>=.5_dp) value=ieee_difference(y(order(index+1)),ieee_product(difference,1-fraction))
    end if
  end function
  real(dp) function log1p_native(x) result(value)
    real(dp), intent(in) :: x
    real(dp) :: z
    z=1+x
    if(z==1) then
      value=x
    else
      value=log(z)-((z-1)-x)/z
    end if
  end function
  real(dp) function matrix_metric(y,predicted,kind,w,alpha,output_weights,raw_values) result(value)
    real(dp), intent(in) :: y(:,:),predicted(:,:)
    integer, intent(in) :: kind
    real(dp), optional, intent(in) :: w(:),alpha,output_weights(:)
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    real(dp) :: values(size(y,2)),baseline(size(y,1)),losses(size(y,1)),a,num,den,difference,mean_square
    integer :: i,j
    a=.5_dp;if(present(alpha)) a=alpha
    call check_alpha(a)
    call check_arrays(y,predicted,w,check_mass=kind/=2.or.size(y,1)>=2)
    if(kind==2.and.size(y,1)<2) then
      values=nan_value();value=aggregate(values,output_weights,raw_values);return
    end if
    if(kind==3) call require(all(y>-1).and.all(predicted>-1),'RMSLE targets must be greater than -1')
    do j=1,size(y,2)
      select case(kind)
      case(1)
        values(j)=pinball(y(:,j),predicted(:,j),a,w)
      case(2)
        num=pinball(y(:,j),predicted(:,j),a,w);baseline=quantile(y(:,j),a,w)
        call require(all(ieee_is_finite(baseline)),'pinball quantile baseline must be finite')
        den=pinball(y(:,j),baseline,a,w);values(j)=1
        if(ieee_is_nan(num)) then
          values(j)=nan_value()
        else if(num/=0) then
          values(j)=0
          if(ieee_is_nan(den)) then
            values(j)=nan_value()
          else if(den/=0) then
            values(j)=ieee_difference(1._dp,ieee_quotient(num,den))
          end if
        end if
      case(3)
        do i=1,size(y,1)
          difference=log1p_native(y(i,j))-log1p_native(predicted(i,j))
          losses(i)=ieee_product(difference,difference)
        end do
        mean_square=ieee_mean(losses,w);values(j)=nan_value()
        if(.not.ieee_is_nan(mean_square)) then
          if(mean_square>=0) values(j)=sqrt(mean_square)
        end if
      end select
    end do
    value=aggregate(values,output_weights,raw_values)
  end function
  real(dp) function quiet_log(x) result(value)
    real(dp), intent(in) :: x
    value=nan_value()
    if(ieee_is_nan(x)) return
    if(x==0) then
      value=ieee_value(0._dp,ieee_negative_inf)
    else if(x>0) then
      value=log(x)
    end if
  end function
  recursive real(dp) function quiet_power(x,exponent) result(value)
    real(dp), intent(in) :: x,exponent
    real(dp) :: logarithm
    value=nan_value()
    if(ieee_is_nan(x).or.ieee_is_nan(exponent)) return
    if(exponent==0) then
      value=1;return
    end if
    if(x<0) then
      if(exponent/=anint(exponent)) return
      value=quiet_power(-x,exponent)
      if(modulo(abs(exponent),2._dp)==1) value=-value
      return
    end if
    if(x==0) then
      value=0
      if(exponent<0) value=ieee_value(0._dp,ieee_positive_inf)
      return
    end if
    logarithm=ieee_product(exponent,quiet_log(x))
    if(ieee_is_nan(logarithm)) return
    if(logarithm>log(huge(1._dp))) then
      value=ieee_value(0._dp,ieee_positive_inf)
    else if(logarithm<log(tiny(1._dp))-40) then
      value=0
    else if(logarithm>=log(huge(1._dp))-1.e-12_dp) then
      value=ieee_product(exp(logarithm-log(2._dp)),2._dp)
    else
      ! Direct power retains source rounding where it is safe.
      value=x**exponent
    end if
  end function
  real(dp) function tweedie_loss(y,predicted,power,w) result(value)
    real(dp), intent(in) :: y(:),predicted(:),power
    real(dp), optional, intent(in) :: w(:)
    real(dp) :: losses(size(y)),ratio,term,first,second,third,base,p1,p2
    integer :: i
    p1=1-power;p2=2-power
    do i=1,size(y)
      if(power==0) then
        term=ieee_difference(y(i),predicted(i));losses(i)=ieee_product(term,term)
      else if(power==1) then
        ratio=ieee_quotient(y(i),predicted(i));term=nan_value()
        if(.not.ieee_is_nan(ratio)) then
          term=0
          if(y(i)/=0) term=ieee_product(y(i),quiet_log(ratio))
        end if
        losses(i)=ieee_product(2._dp,ieee_difference(ieee_difference(term,y(i)),-predicted(i)))
      else if(power==2) then
        first=quiet_log(ieee_quotient(predicted(i),y(i)));second=ieee_quotient(y(i),predicted(i))
        losses(i)=ieee_product(2._dp,ieee_difference(ieee_difference(first,-second),1._dp))
      else
        base=y(i);if(power<0) base=max(base,0._dp)
        first=ieee_quotient(quiet_power(base,p2),ieee_product(p1,p2))
        second=ieee_quotient(ieee_product(y(i),quiet_power(predicted(i),p1)),p1)
        third=ieee_quotient(quiet_power(predicted(i),p2),p2)
        losses(i)=ieee_product(2._dp,ieee_difference(ieee_difference(first,second),-third))
      end if
    end do
    value=ieee_mean(losses,w)
  end function
  subroutine check_power(power)
    real(dp), intent(in) :: power
    call require(ieee_is_finite(power),'Tweedie power must be finite')
    call require(power<=0.or.power>=1,'Tweedie power must be at most zero or at least one')
  end subroutine
  subroutine check_tweedie_domain(y,predicted,power)
    real(dp), intent(in) :: y(:),predicted(:),power
    if(power/=0) call require(all(predicted>0),'Tweedie predictions must be strictly positive')
    if(power>=1.and.power<2) call require(all(y>=0),'Tweedie targets must be nonnegative')
    if(power>=2) call require(all(y>0),'Tweedie targets must be strictly positive')
  end subroutine
  real(dp) function mean_tweedie_deviance(y,predicted,sample_weight,power) result(value)
    real(dp), intent(in) :: y(:),predicted(:)
    real(dp), optional, intent(in) :: sample_weight(:),power
    real(dp) :: p
    p=0;if(present(power)) p=power
    call check_power(p)
    call check_arrays(reshape(y,[size(y),1]),reshape(predicted,[size(predicted),1]),sample_weight)
    call check_tweedie_domain(y,predicted,p)
    value=tweedie_loss(y,predicted,p,sample_weight)
  end function
  real(dp) function d2_tweedie_score(y,predicted,sample_weight,power) result(value)
    real(dp), intent(in) :: y(:),predicted(:)
    real(dp), optional, intent(in) :: sample_weight(:),power
    real(dp) :: p,num,den,baseline(size(y))
    p=0;if(present(power)) p=power
    call check_power(p)
    call check_arrays(reshape(y,[size(y),1]),reshape(predicted,[size(predicted),1]),sample_weight, &
      check_mass=size(y)>=2)
    if(size(y)<2) then
      value=nan_value();return
    end if
    call check_tweedie_domain(y,predicted,p)
    num=tweedie_loss(y,predicted,p,sample_weight);baseline=ieee_mean(y,sample_weight)
    den=tweedie_loss(y,baseline,p,sample_weight)
    if(.not.ieee_is_nan(den)) call require(den/=0,'D2 Tweedie null deviance is zero')
    value=ieee_difference(1._dp,ieee_quotient(num,den))
  end function
  real(dp) function mean_pinball_loss_vector(y,predicted,sample_weight,alpha,output_weights,raw_values) result(value)
    real(dp), intent(in) :: y(:),predicted(:)
    real(dp), optional, intent(in) :: sample_weight(:),output_weights(:)
    real(dp), optional, intent(in) :: alpha
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    value=matrix_metric(reshape(y,[size(y),1]),reshape(predicted,[size(predicted),1]),1,sample_weight,alpha,output_weights,raw_values)
  end function
  real(dp) function mean_pinball_loss_matrix(y,predicted,sample_weight,alpha,output_weights,raw_values) result(value)
    real(dp), intent(in) :: y(:,:),predicted(:,:)
    real(dp), optional, intent(in) :: sample_weight(:),output_weights(:)
    real(dp), optional, intent(in) :: alpha
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    value=matrix_metric(y,predicted,1,sample_weight,alpha,output_weights,raw_values)
  end function
  real(dp) function d2_pinball_score_vector(y,predicted,sample_weight,alpha,output_weights,raw_values) result(value)
    real(dp), intent(in) :: y(:),predicted(:)
    real(dp), optional, intent(in) :: sample_weight(:),output_weights(:)
    real(dp), optional, intent(in) :: alpha
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    value=matrix_metric(reshape(y,[size(y),1]),reshape(predicted,[size(predicted),1]),2,sample_weight,alpha,output_weights,raw_values)
  end function
  real(dp) function d2_pinball_score_matrix(y,predicted,sample_weight,alpha,output_weights,raw_values) result(value)
    real(dp), intent(in) :: y(:,:),predicted(:,:)
    real(dp), optional, intent(in) :: sample_weight(:),output_weights(:)
    real(dp), optional, intent(in) :: alpha
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    value=matrix_metric(y,predicted,2,sample_weight,alpha,output_weights,raw_values)
  end function
  real(dp) function d2_absolute_error_score_vector(y,predicted,sample_weight,output_weights,raw_values) result(value)
    real(dp), intent(in) :: y(:),predicted(:)
    real(dp), optional, intent(in) :: sample_weight(:),output_weights(:)
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    value=matrix_metric(reshape(y,[size(y),1]),reshape(predicted,[size(predicted),1]),2,sample_weight,.5_dp,output_weights,raw_values)
  end function
  real(dp) function d2_absolute_error_score_matrix(y,predicted,sample_weight,output_weights,raw_values) result(value)
    real(dp), intent(in) :: y(:,:),predicted(:,:)
    real(dp), optional, intent(in) :: sample_weight(:),output_weights(:)
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    value=matrix_metric(y,predicted,2,sample_weight,.5_dp,output_weights,raw_values)
  end function
  real(dp) function root_mean_squared_log_error_vector(y,predicted,sample_weight,output_weights,raw_values) result(value)
    real(dp), intent(in) :: y(:),predicted(:)
    real(dp), optional, intent(in) :: sample_weight(:),output_weights(:)
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    value=matrix_metric(reshape(y,[size(y),1]),reshape(predicted,[size(predicted),1]),3,sample_weight,.5_dp,output_weights,raw_values)
  end function
  real(dp) function root_mean_squared_log_error_matrix(y,predicted,sample_weight,output_weights,raw_values) result(value)
    real(dp), intent(in) :: y(:,:),predicted(:,:)
    real(dp), optional, intent(in) :: sample_weight(:),output_weights(:)
    real(dp), allocatable, optional, intent(out) :: raw_values(:)
    value=matrix_metric(y,predicted,3,sample_weight,.5_dp,output_weights,raw_values)
  end function
end module
