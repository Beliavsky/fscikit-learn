! Binary ranking, calibration and Brier contracts: sklearn 1.6.1
! metrics/_ranking.py, metrics/_classification.py, calibration.py (BSD-3-Clause).
! NumPy 2.2.6 pairwise sums and percentile interpolation (BSD-3-Clause).
! Complete upstream notices: THIRD_PARTY_NOTICES.md and LICENSES/.
module sklearn_ranking
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite,ieee_is_nan,ieee_value,ieee_quiet_nan,ieee_positive_inf
  implicit none
  private
  public :: roc_curve,precision_recall_curve,det_curve,auc,roc_auc_score,average_precision_score
  public :: calibration_curve,brier_score_loss
  public :: stable_order ! Internal shared stable merge-sort; not reexported by sklearn.
contains
  real(dp) function undefined() result(value)
    value=ieee_value(1._dp,ieee_quiet_nan)
  end function
  elemental real(dp) function ratio(a,b) result(value)
    real(dp), intent(in) :: a,b
    if(b==0) then
      value=ieee_value(1._dp,ieee_quiet_nan)
      if(a/=0) value=sign(ieee_value(1._dp,ieee_positive_inf),a)*sign(1._dp,b)
    else if(abs(b)<1) then
      if(abs(a)>huge(1._dp)*abs(b)) then
        value=sign(ieee_value(1._dp,ieee_positive_inf),a)*sign(1._dp,b)
      else
        value=a/b
      end if
    else
      value=a/b
    end if
  end function
  recursive real(dp) function pair_sum(values) result(total)
    real(dp), intent(in) :: values(:)
    real(dp) :: partial(8)
    integer :: n,i,whole,split
    n=size(values)
    if(n<8) then
      total=-0._dp
      do i=1,n
        total=total+values(i)
      end do
    else if(n<=128) then
      partial=values(:8); whole=n-mod(n,8)
      do i=9,whole,8
        partial=partial+values(i:i+7)
      end do
      total=((partial(1)+partial(2))+(partial(3)+partial(4))) &
        +((partial(5)+partial(6))+(partial(7)+partial(8)))
      do i=whole+1,n
        total=total+values(i)
      end do
    else
      split=n/2; split=split-mod(split,8)
      total=pair_sum(values(:split))+pair_sum(values(split+1:))
    end if
  end function
  function stable_order(values) result(order)
    real(dp), intent(in) :: values(:)
    integer :: order(size(values)),work(size(values)),n,width,left,middle,right,i,j,out
    n=size(values); order=[(i,i=1,n)]; width=1
    do while(width<n)
      left=1
      do while(left<=n)
        middle=left-1+min(width,n-left+1); right=middle+min(width,n-middle)
        i=left; j=middle+1; out=left
        do while(i<=middle.and.j<=right)
          if(values(order(i))<=values(order(j))) then
            work(out)=order(i); i=i+1
          else
            work(out)=order(j); j=j+1
          end if
          out=out+1
        end do
        do while(i<=middle)
          work(out)=order(i); i=i+1; out=out+1
        end do
        do while(j<=right)
          work(out)=order(j); j=j+1; out=out+1
        end do
        left=right+1
      end do
      order=work
      if(width>n/2) exit
      width=2*width
    end do
  end function
  integer function label_count(y) result(classes)
    integer, intent(in) :: y(:)
    integer :: i,first,second
    classes=1; first=y(1); second=first
    do i=2,size(y)
      if(y(i)==first.or.y(i)==second) cycle
      if(classes==2) then
        classes=3; return
      end if
      second=y(i); classes=2
    end do
  end function
  integer function positive_label(y,pos_label) result(pos)
    integer, intent(in) :: y(:)
    integer, intent(in), optional :: pos_label
    pos=1
    if(present(pos_label)) then
      pos=pos_label; return
    end if
    call require(all(y==0.or.y==1).or.all(y==-1.or.y==1),'ranking requires an explicit positive label')
  end function
  subroutine check_vectors(y,scores)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    call require(size(y,kind=int64)<huge(1),'ranking sample count exceeds native index range')
    call require(size(y)>0.and.size(y)==size(scores),'ranking sample count mismatch or empty input')
    call require(all(ieee_is_finite(scores)),'ranking scores must be finite')
  end subroutine
  function sample_weights(n,sample_weight) result(weights)
    integer, intent(in) :: n
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp) :: weights(n),mass
    integer :: i
    weights=1
    if(.not.present(sample_weight)) return
    call require(size(sample_weight)==n,'ranking sample weight length mismatch')
    call require(all(ieee_is_finite(sample_weight)),'ranking sample weights must be finite')
    mass=0
    do i=1,n
      call require(abs(sample_weight(i))<=huge(1._dp)/8-mass,'ranking weight mass exceeds working range')
      mass=mass+abs(sample_weight(i))
    end do
    weights=sample_weight
  end function
  subroutine binary_curve(y,scores,fps,tps,thresholds,pos_label,sample_weight)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    real(dp), allocatable, intent(out) :: fps(:),tps(:),thresholds(:)
    integer, intent(in), optional :: pos_label
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: weights(:),filtered(:),fp(:),tp(:),cutoffs(:)
    integer, allocatable :: active(:),order(:)
    real(dp) :: positives,negatives
    integer :: n,i,j,m,pos
    logical :: last
    call check_vectors(y,scores)
    if(.not.present(pos_label)) call require(label_count(y)<=2,'ranking multiclass targets need an explicit positive label')
    weights=sample_weights(size(y),sample_weight)
    active=pack([(i,i=1,size(y))],weights/=0)
    call require(size(active)>0,'ranking needs at least one nonzero-weight sample')
    pos=positive_label(y(active),pos_label); filtered=scores(active); order=stable_order(filtered)
    n=size(active); active=active(order(n:1:-1)); allocate(fp(n),tp(n),cutoffs(n))
    positives=0; negatives=0; m=0
    do i=1,n
      j=active(i)
      if(y(j)==pos) then
        positives=positives+weights(j)
      else
        negatives=negatives+weights(j)
      end if
      last=i==n
      if(i<n) last=scores(j)/=scores(active(i+1))
      if(.not.last) cycle
      m=m+1; tp(m)=positives; fp(m)=negatives
      if(.not.present(sample_weight)) fp(m)=real(i,dp)-positives
      cutoffs(m)=scores(j)
    end do
    fps=fp(:m); tps=tp(:m); thresholds=cutoffs(:m)
  end subroutine
  subroutine roc_curve(y,scores,fpr,tpr,thresholds,pos_label,sample_weight,drop_intermediate)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    real(dp), allocatable, intent(out) :: fpr(:),tpr(:),thresholds(:)
    integer, intent(in), optional :: pos_label
    real(dp), intent(in), optional :: sample_weight(:)
    logical, intent(in), optional :: drop_intermediate
    real(dp), allocatable :: fp(:),tp(:),cuts(:)
    logical, allocatable :: keep(:)
    logical :: drop
    integer :: i,n
    call binary_curve(y,scores,fp,tp,cuts,pos_label,sample_weight)
    n=size(fp); drop=.true.; if(present(drop_intermediate)) drop=drop_intermediate
    if(drop.and.n>2) then
      allocate(keep(n)); keep=.true.
      do i=2,n-1
        keep(i)=((fp(i+1)-fp(i))-(fp(i)-fp(i-1)))/=0.or.((tp(i+1)-tp(i))-(tp(i)-tp(i-1)))/=0
      end do
      fp=pack(fp,keep); tp=pack(tp,keep); cuts=pack(cuts,keep)
    end if
    fp=[0._dp,fp]; tp=[0._dp,tp]; thresholds=[ieee_value(1._dp,ieee_positive_inf),cuts]
    n=size(fp); fpr=spread(undefined(),1,n); tpr=fpr
    if(fp(n)>0) fpr=ratio(fp,fp(n))
    if(tp(n)>0) tpr=ratio(tp,tp(n))
  end subroutine
  subroutine precision_recall_curve(y,scores,precision,recall,thresholds,pos_label,sample_weight,drop_intermediate)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    real(dp), allocatable, intent(out) :: precision(:),recall(:),thresholds(:)
    integer, intent(in), optional :: pos_label
    real(dp), intent(in), optional :: sample_weight(:)
    logical, intent(in), optional :: drop_intermediate
    real(dp), allocatable :: fp(:),tp(:),cuts(:),p(:),r(:)
    logical, allocatable :: keep(:)
    logical :: drop
    integer :: i,n
    call binary_curve(y,scores,fp,tp,cuts,pos_label,sample_weight)
    n=size(fp); drop=.false.; if(present(drop_intermediate)) drop=drop_intermediate
    if(drop.and.n>2) then
      allocate(keep(n)); keep=.true.
      do i=2,n-1
        keep(i)=tp(i)/=tp(i-1).or.tp(i+1)/=tp(i)
      end do
      fp=pack(fp,keep); tp=pack(tp,keep); cuts=pack(cuts,keep)
    end if
    n=size(fp); allocate(p(n),r(n)); p=0; r=1
    do i=1,n
      if(tp(i)+fp(i)/=0) p(i)=ratio(tp(i),tp(i)+fp(i))
    end do
    if(tp(n)/=0) r=ratio(tp,tp(n))
    precision=[p(n:1:-1),1._dp]; recall=[r(n:1:-1),0._dp]; thresholds=cuts(n:1:-1)
  end subroutine
  integer function insertion(values,key,right_side) result(index)
    real(dp), intent(in) :: values(:),key
    logical, intent(in) :: right_side
    integer :: first,last,middle
    first=1; last=size(values)+1
    do while(first<last)
      middle=first+(last-first)/2
      if(values(middle)<key.or.(right_side.and.values(middle)==key)) then
        first=middle+1
      else
        last=middle
      end if
    end do
    index=first
  end function
  subroutine det_curve(y,scores,fpr,fnr,thresholds,pos_label,sample_weight)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    real(dp), allocatable, intent(out) :: fpr(:),fnr(:),thresholds(:)
    integer, intent(in), optional :: pos_label
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: fp(:),tp(:),cuts(:),fn(:)
    integer :: n,first,last
    call binary_curve(y,scores,fp,tp,cuts,pos_label,sample_weight)
    call require(label_count(y)==2,'DET curve requires exactly two classes')
    n=size(fp); fn=tp(n)-tp; first=insertion(fp,fp(1),.true.)-1; last=insertion(tp,tp(n),.false.)
    fpr=ratio(fp(last:first:-1),fp(n)); fnr=ratio(fn(last:first:-1),tp(n)); thresholds=cuts(last:first:-1)
  end subroutine
  real(dp) function auc(x,y) result(area)
    real(dp), intent(in) :: x(:),y(:)
    real(dp), allocatable :: dx(:),terms(:)
    real(dp) :: direction
    integer :: n
    n=size(x)
    call require(n==size(y).and.n>=2,'AUC requires equal vectors with at least two points')
    call require(all(ieee_is_finite(x)).and.all(ieee_is_finite(y)),'AUC coordinates must be finite')
    dx=x(2:)-x(:n-1); direction=1
    if(any(dx<0)) then
      call require(all(dx<=0),'AUC x coordinates must be monotonic'); direction=-1
    end if
    terms=dx*(y(:n-1)+y(2:))/2; area=direction*pair_sum(terms)
  end function
  real(dp) function roc_auc_score(y,scores,pos_label,sample_weight,max_fpr) result(area)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    integer, intent(in), optional :: pos_label
    real(dp), intent(in), optional :: sample_weight(:),max_fpr
    real(dp), allocatable :: fp(:),tp(:),thresholds(:),weights(:)
    real(dp) :: limit,interpolated,minimum
    integer :: pos,stop
    call check_vectors(y,scores); weights=sample_weights(size(y),sample_weight)
    limit=1; if(present(max_fpr)) limit=max_fpr
    call require(ieee_is_finite(limit),'max_fpr must be finite')
    call require(limit>0.and.limit<=1,'max_fpr must be in (0,1]')
    call require(label_count(y)<=2,'ROC AUC requires binary targets')
    area=undefined(); if(label_count(y)==1) return
    pos=maxval(y); if(present(pos_label)) pos=pos_label
    call require(any(y==pos),'invalid positive label')
    call roc_curve(y,scores,fp,tp,thresholds,pos,sample_weight)
    if(limit==1) then
      if(any(ieee_is_nan(fp)).or.any(ieee_is_nan(tp))) return
      area=auc(fp,tp); return
    end if
    call require(.not.any(ieee_is_nan(fp)),'partial ROC AUC requires negative-class support')
    stop=insertion(fp,limit,.true.)
    call require(stop>=2,'partial ROC AUC requires negative-class support')
    if(any(ieee_is_nan(fp)).or.any(ieee_is_nan(tp))) return
    interpolated=tp(stop-1)+(tp(stop)-tp(stop-1))*(limit-fp(stop-1))/(fp(stop)-fp(stop-1))
    area=auc([fp(:stop-1),limit],[tp(:stop-1),interpolated])
    minimum=.5_dp*limit**2; area=.5_dp*(1+(area-minimum)/(limit-minimum))
  end function
  real(dp) function average_precision_score(y,scores,pos_label,sample_weight) result(area)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: scores(:)
    integer, intent(in), optional :: pos_label
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: precision(:),recall(:),thresholds(:)
    integer :: pos,n
    call check_vectors(y,scores); call require(label_count(y)<=2,'average precision requires binary targets')
    pos=1; if(present(pos_label)) pos=pos_label
    if(label_count(y)==2) call require(any(y==pos),'invalid average precision positive label')
    call precision_recall_curve(y,scores,precision,recall,thresholds,pos,sample_weight)
    n=size(recall); area=-pair_sum((recall(2:)-recall(:n-1))*precision(:n-1))
    if(area<=0) area=0
  end function
  subroutine calibration_curve(y,prob,prob_true,prob_pred,n_bins,strategy,pos_label)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:)
    real(dp), allocatable, intent(out) :: prob_true(:),prob_pred(:)
    integer, intent(in), optional :: n_bins,pos_label
    character(*), intent(in), optional :: strategy
    integer, allocatable :: order(:),counts(:),positives(:)
    real(dp), allocatable :: edges(:),sums(:),sorted(:)
    real(dp) :: step,q,index,fraction,difference
    integer :: bins,pos,i,j,left,right
    character(:), allocatable :: selected
    call check_vectors(y,prob); pos=positive_label(y,pos_label)
    call require(label_count(y)<=2,'calibration curve requires binary targets')
    call require(all(prob>=0).and.all(prob<=1),'calibration probabilities must lie in [0,1]')
    bins=5; if(present(n_bins)) bins=n_bins
    call require(bins>0.and.bins<huge(bins),'calibration n_bins must be positive and below native index limit')
    selected='uniform'; if(present(strategy)) selected=trim(strategy)
    call require(selected=='uniform'.or.selected=='quantile','unknown calibration bin strategy')
    allocate(edges(bins+1),sums(bins),counts(bins),positives(bins)); sums=0; counts=0; positives=0
    step=1._dp/bins; edges=[(i*step,i=0,bins)]; edges(bins+1)=1
    if(selected=='quantile') then
      order=stable_order(prob); sorted=prob(order)
      do i=1,bins+1
        q=(edges(i)*100)/100; index=(size(prob)-1)*q; left=floor(index)+1; right=min(size(prob),left+1)
        fraction=index-floor(index); difference=sorted(right)-sorted(left)
        if(fraction<.5_dp) then
          edges(i)=sorted(left)+difference*fraction
        else
          edges(i)=sorted(right)-difference*(1-fraction)
        end if
      end do
    end if
    do i=1,size(y)
      j=insertion(edges(2:bins),prob(i),.false.); counts(j)=counts(j)+1; sums(j)=sums(j)+prob(i)
      if(y(i)==pos) positives(j)=positives(j)+1
    end do
    prob_true=real(pack(positives,counts>0),dp)/real(pack(counts,counts>0),dp)
    prob_pred=pack(sums,counts>0)/real(pack(counts,counts>0),dp)
  end subroutine
  real(dp) function brier_score_loss(y,prob,pos_label,sample_weight) result(loss)
    integer, intent(in) :: y(:)
    real(dp), intent(in) :: prob(:)
    integer, intent(in), optional :: pos_label
    real(dp), intent(in), optional :: sample_weight(:)
    real(dp), allocatable :: weights(:),errors(:)
    real(dp) :: mass
    integer :: pos
    call check_vectors(y,prob); call require(label_count(y)<=2,'Brier loss requires binary targets')
    call require(all(prob>=0).and.all(prob<=1),'Brier probabilities must lie in [0,1]')
    pos=maxval(y)
    if(present(pos_label)) then
      pos=pos_label
    else if(all(y==0.or.y==1).or.all(y==-1.or.y==1)) then
      pos=1
    end if
    weights=sample_weights(size(y),sample_weight); mass=pair_sum(weights)
    call require(mass/=0,'Brier sample weights must have nonzero sum')
    errors=(merge(1._dp,0._dp,y==pos)-prob)**2; loss=ratio(pair_sum(weights*errors),mass)
  end function
end module
