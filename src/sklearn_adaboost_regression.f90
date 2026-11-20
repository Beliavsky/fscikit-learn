module sklearn_adaboost_regression
  use sklearn_common, only: dp,require,check_x,check_xy,argsort
  use sklearn_tree, only: decision_tree_regressor
  use sklearn_metrics, only: r2_score
  use sklearn_legacy_rng, only: legacy_random_state
  use iso_fortran_env, only: int64,real32
  use ieee_arithmetic, only: ieee_is_finite
  implicit none
  private
  public :: adaboost_regressor
  ! AdaBoost.R2 and dense squared-error CART, following sklearn 1.6.1.
  type :: adaboost_regressor
    integer :: n_estimators=50,n_estimators_=0,n_features_in_=0,random_state=0
    integer :: max_depth=3,min_samples_split=2,min_samples_leaf=1,max_features=0
    real(dp) :: learning_rate=1,min_impurity_decrease=0
    character(16) :: loss='linear'
    type(decision_tree_regressor), allocatable :: estimators_(:)
    real(dp), allocatable :: estimator_weights_(:),estimator_errors_(:),feature_importances_(:)
  contains
    procedure :: fit
    procedure :: predict
    procedure :: staged_predict
    procedure :: score
    procedure :: staged_score
  end type
contains
  function float_features(x) result(z)
    real(dp), intent(in) :: x(:,:)
    real(dp) :: z(size(x,1),size(x,2))
    call check_x(x)
    call require(maxval(abs(x))<=real(huge(0._real32),dp),'AdaBoost features exceed float32 range')
    z=real(real(x,real32),dp)
  end function
  subroutine fit(self,x,y,sample_weight)
    class(adaboost_regressor), intent(inout) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), optional, intent(in) :: sample_weight(:)
    type(legacy_random_state) :: rng
    real(dp) :: z(size(x,1),size(x,2)),w(size(y)),cdf(size(y)),err(size(y)),v,e,beta,total,u
    logical :: positive(size(y))
    integer :: i,j,k,n,seed,ids(size(y))
    call check_xy(x,size(y)); call check_x(reshape(y,[size(y),1]))
    call require(self%n_estimators>=1.and.self%random_state>=0,'invalid AdaBoost estimator count or seed')
    call require(ieee_is_finite(self%learning_rate).and.self%learning_rate>0,'invalid AdaBoost learning rate')
    call require(self%loss=='linear'.or.self%loss=='square'.or.self%loss=='exponential','invalid AdaBoost loss')
    call require(self%max_depth>=1.and.self%min_samples_split>=2.and.self%min_samples_leaf>=1,'invalid CART size')
    call require(self%max_features>=0.and.self%max_features<=size(x,2),'invalid CART max_features')
    call require(ieee_is_finite(self%min_impurity_decrease).and.self%min_impurity_decrease>=0,'invalid impurity decrease')
    n=size(y); z=float_features(x); w=1
    if(present(sample_weight)) then
      call require(size(sample_weight)==n,'sample weight shape mismatch')
      call require(all(ieee_is_finite(sample_weight)).and.all(sample_weight>=0),'invalid sample weights')
      w=sample_weight
    end if
    total=sum(w)
    call require(ieee_is_finite(total).and.total>0,'sample weights must have positive finite sum')
    w=w/total; positive=w>0
    if(allocated(self%estimators_)) deallocate(self%estimators_)
    allocate(self%estimators_(self%n_estimators))
    self%estimator_weights_=spread(0._dp,1,self%n_estimators)
    self%estimator_errors_=spread(1._dp,1,self%n_estimators)
    self%feature_importances_=spread(0._dp,1,size(x,2))
    self%n_features_in_=size(x,2); self%n_estimators_=0
    call rng%seed(self%random_state)
    do k=1,self%n_estimators
      where(positive) w=max(w,epsilon(1._dp))
      seed=rng%integer_index(2147483647)-1
      cdf(1)=w(1)
      do i=2,n
        cdf(i)=cdf(i-1)+w(i)
      end do
      cdf=cdf/cdf(n)
      do i=1,n
        u=rng%uniform(); j=1
        do while(j<n)
          if(u<cdf(j)) exit
          j=j+1
        end do
        ids(i)=j
      end do
      self%estimators_(k)%max_depth=self%max_depth
      self%estimators_(k)%min_samples_split=self%min_samples_split
      self%estimators_(k)%min_samples_leaf=self%min_samples_leaf
      self%estimators_(k)%max_features=self%max_features
      self%estimators_(k)%min_impurity_decrease=self%min_impurity_decrease
      self%estimators_(k)%random_state=seed
      call cart(self%estimators_(k),z(ids,:),y(ids))
      err=abs(self%estimators_(k)%predict(z)-y); v=maxval(err,mask=positive)
      where(.not.positive) err=0
      if(v>0) err=err/v
      select case(trim(self%loss))
      case('square'); err=err**2
      case('exponential'); err=1-exp(-err)
      end select
      e=sum(w*err)
      if(e>=.5_dp) then
        if(k==1) self%n_estimators_=1
        exit
      end if
      self%n_estimators_=k
      self%estimator_errors_(k)=e
      if(e<=0) then
        self%estimator_weights_(k)=1
        exit
      end if
      beta=e/(1-e); self%estimator_weights_(k)=self%learning_rate*log(1/beta)
      if(k<self%n_estimators) then
        where(positive) w=w*beta**((1-err)*self%learning_rate)
      end if
      total=sum(w)
      if(.not.ieee_is_finite(total).or.total<=0) exit
      if(k<self%n_estimators) w=w/total
    end do
    self%estimators_=self%estimators_(:self%n_estimators_)
    total=sum(self%estimator_weights_)
    if(total>0) then
      do k=1,self%n_estimators_
        self%feature_importances_=self%feature_importances_+ &
          self%estimator_weights_(k)*self%estimators_(k)%feature_importances_/total
      end do
    end if
  end subroutine
  function staged_predict(self,x) result(out)
    class(adaboost_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp), allocatable :: out(:,:),values(:,:),z(:,:)
    real(dp) :: cumulative,total
    integer, allocatable :: order(:)
    integer :: i,j,k
    call require(allocated(self%estimators_),'AdaBoostRegressor is not fitted')
    call require(size(x,2)==self%n_features_in_,'AdaBoost feature count mismatch')
    z=float_features(x)
    allocate(values(size(x,1),self%n_estimators_),out(size(x,1),self%n_estimators_))
    do k=1,self%n_estimators_
      values(:,k)=self%estimators_(k)%predict(z)
      do i=1,size(x,1)
        order=argsort(values(i,:k)); total=sum(self%estimator_weights_(order)); cumulative=0
        do j=1,k
          cumulative=cumulative+self%estimator_weights_(order(j))
          if(cumulative>=.5_dp*total) exit
        end do
        out(i,k)=values(i,order(j))
      end do
    end do
  end function
  function predict(self,x) result(out)
    class(adaboost_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:)
    real(dp) :: out(size(x,1))
    real(dp), allocatable :: stages(:,:)
    stages=self%staged_predict(x); out=stages(:,self%n_estimators_)
  end function
  real(dp) function score(self,x,y) result(v)
    class(adaboost_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    v=r2_score(y,self%predict(x))
  end function
  function staged_score(self,x,y) result(v)
    class(adaboost_regressor), intent(in) :: self
    real(dp), intent(in) :: x(:,:),y(:)
    real(dp), allocatable :: v(:),stages(:,:)
    integer :: k
    stages=self%staged_predict(x); allocate(v(self%n_estimators_))
    do k=1,size(v)
      v(k)=r2_score(y,stages(:,k))
    end do
  end function
  subroutine cart(tree,x,y)
    type(decision_tree_regressor), intent(inout) :: tree
    real(dp), intent(in) :: x(:,:),y(:)
    type(legacy_random_state) :: rng
    integer :: samples(size(y)),features(size(x,2)),constants(size(x,2)),i,root
    integer(int64) :: state
    call rng%seed(tree%random_state)
    state=int(rng%integer_index(2147483647)-1,int64)
    tree%n_features_in_=size(x,2); tree%node_count_=0
    allocate(tree%nodes_(2*size(y)))
    tree%feature_importances_=spread(0._dp,1,size(x,2))
    samples=[(i,i=1,size(y))]; features=[(i,i=1,size(x,2))]; constants=0
    root=grow(1,size(y),0,0,0._dp)
    if(sum(tree%feature_importances_)>0) &
      tree%feature_importances_=tree%feature_importances_/sum(tree%feature_importances_)
  contains
    integer function draw(lo,hi) result(v)
      integer, intent(in) :: lo,hi
      if(state==0) state=1
      state=iand(ieor(state,ishft(state,13)),4294967295_int64)
      state=ieor(state,ishft(state,-17))
      state=iand(ieor(state,ishft(state,5)),4294967295_int64)
      v=lo+int(modulo(iand(state,2147483647_int64),int(hi-lo,int64)))
    end function
    subroutine swap(a,b)
      integer, intent(inout) :: a,b
      integer :: tmp
      tmp=a; a=b; b=tmp
    end subroutine
    ! Same median-of-three, three-way introsort and heap fallback as DensePartitioner.
    recursive subroutine sort_samples(offset,length,maxdepth,feature)
      integer, intent(in) :: offset,length,maxdepth,feature
      integer :: base,n,remaining,j,left,right
      real(dp) :: a,b,c,pivot
      base=offset; n=length; remaining=maxdepth
      do while(n>1)
        if(remaining<=0) then
          call heap_sort(base,n,feature)
          return
        end if
        remaining=remaining-1
        a=x(samples(base),feature); b=x(samples(base+n/2),feature); c=x(samples(base+n-1),feature)
        if(a<b) then
          if(b<c) then
            pivot=b
          else if(a<c) then
            pivot=c
          else
            pivot=a
          end if
        else if(b<c) then
          if(a<c) then
            pivot=a
          else
            pivot=c
          end if
        else
          pivot=b
        end if
        j=0; left=0; right=n
        do while(j<right)
          if(x(samples(base+j),feature)<pivot) then
            call swap(samples(base+j),samples(base+left)); j=j+1; left=left+1
          else if(x(samples(base+j),feature)>pivot) then
            right=right-1; call swap(samples(base+j),samples(base+right))
          else
            j=j+1
          end if
        end do
        call sort_samples(base,left,remaining,feature)
        base=base+right; n=n-right
      end do
    end subroutine
    subroutine heap_sort(base,n,feature)
      integer, intent(in) :: base,n,feature
      integer :: start,last
      start=(n-2)/2
      do
        call sift(base,start,n,feature)
        if(start==0) exit
        start=start-1
      end do
      do last=n-1,1,-1
        call swap(samples(base),samples(base+last)); call sift(base,0,last,feature)
      end do
    end subroutine
    subroutine sift(base,start,last,feature)
      integer, intent(in) :: base,start,last,feature
      integer :: root,child,largest
      root=start
      do
        child=root*2+1; largest=root
        if(child<last) then
          if(x(samples(base+largest),feature)<x(samples(base+child),feature)) largest=child
        end if
        if(child+1<last) then
          if(x(samples(base+largest),feature)<x(samples(base+child+1),feature)) largest=child+1
        end if
        if(largest==root) exit
        call swap(samples(base+root),samples(base+largest)); root=largest
      end do
    end subroutine
    recursive integer function grow(first,last,depth,known,inherited) result(node)
      integer, intent(in) :: first,last,depth,known
      real(dp), intent(in) :: inherited
      integer :: n,p,a,b,j,k,f,fi,fj,found,drawn,nconstant,visited,nf,cut,bestcut,bestfeature,pos
      integer :: leftnode,rightnode
      real(dp) :: total,squares,parent,leftsum,leftsq,il,ir,proxy,best,threshold,bestthreshold,gain
      n=last-first+1; p=size(x,2); total=0; squares=0
      do a=first,last
        total=total+y(samples(a)); squares=squares+y(samples(a))**2
      end do
      parent=inherited
      if(depth==0) parent=squares/n-(total/n)**2
      tree%node_count_=tree%node_count_+1; node=tree%node_count_
      tree%nodes_(node)%n_samples=n; tree%nodes_(node)%depth=depth
      tree%nodes_(node)%value=total/n; tree%nodes_(node)%impurity=parent
      if(depth>=tree%max_depth.or.n<max(tree%min_samples_split,2*tree%min_samples_leaf)) return
      if(parent<=epsilon(1._dp)) return
      fi=p; found=0; drawn=0; nconstant=known; visited=0; nf=tree%max_features
      if(nf==0) nf=p
      best=-huge(1._dp); bestfeature=0; bestcut=0; bestthreshold=0
      do while(fi>nconstant.and.(visited<nf.or.visited<=found+drawn))
        visited=visited+1; fj=draw(drawn,fi-found)
        if(fj<known) then
          call swap(features(drawn+1),features(fj+1)); drawn=drawn+1
          cycle
        end if
        fj=fj+found; f=features(fj+1)
        call sort_samples(first,n,2*int(log(real(n,dp))/log(2._dp)),f)
        if(x(samples(last),f)<=real(real(x(samples(first),f),real32)+1.e-7_real32,dp)) then
          call swap(features(fj+1),features(nconstant+1)); found=found+1; nconstant=nconstant+1
          cycle
        end if
        fi=fi-1; call swap(features(fi+1),features(fj+1)); leftsum=0; pos=0
        do k=1,n-1
          if(x(samples(first+k),f)<=real(real(x(samples(first+k-1),f),real32)+1.e-7_real32,dp)) cycle
          if(k<tree%min_samples_leaf.or.n-k<tree%min_samples_leaf) cycle
          if(k-pos<=n-k) then
            do a=pos+1,k
              leftsum=leftsum+y(samples(first+a-1))
            end do
          else
            leftsum=total
            do a=n,k+1,-1
              leftsum=leftsum-y(samples(first+a-1))
            end do
          end if
          pos=k; proxy=leftsum**2/k+(total-leftsum)**2/(n-k)
          if(proxy>best) then
            best=proxy; bestfeature=f; bestcut=k
            threshold=x(samples(first+k-1),f)/2+x(samples(first+k),f)/2
            if(threshold==x(samples(first+k),f)) threshold=x(samples(first+k-1),f)
            bestthreshold=threshold
          end if
        end do
      end do
      features(:known)=constants(:known); constants(known+1:nconstant)=features(known+1:nconstant)
      if(bestfeature==0) return
      a=first; b=last
      do while(a<b)
        if(x(samples(a),bestfeature)<=bestthreshold) then
          a=a+1
        else
          call swap(samples(a),samples(b)); b=b-1
        end if
      end do
      cut=first+bestcut-1; leftsum=0; leftsq=0
      do a=first,cut
        leftsq=leftsq+y(samples(a))**2
      end do
      if(bestcut<=n-bestcut) then
        do a=first,cut
          leftsum=leftsum+y(samples(a))
        end do
      else
        leftsum=total
        do a=last,cut+1,-1
          leftsum=leftsum-y(samples(a))
        end do
      end if
      il=leftsq/bestcut-(leftsum/bestcut)**2
      ir=(squares-leftsq)/(n-bestcut)-((total-leftsum)/(n-bestcut))**2
      gain=n*(parent-real(bestcut,dp)/n*il-real(n-bestcut,dp)/n*ir)
      if(gain/size(y)+epsilon(1._dp)<tree%min_impurity_decrease) return
      tree%nodes_(node)%feature=bestfeature; tree%nodes_(node)%threshold=bestthreshold
      tree%feature_importances_(bestfeature)=tree%feature_importances_(bestfeature)+gain
      leftnode=grow(first,cut,depth+1,nconstant,il)
      rightnode=grow(cut+1,last,depth+1,nconstant,ir)
      tree%nodes_(node)%left=leftnode; tree%nodes_(node)%right=rightnode
    end function
  end subroutine
end module
