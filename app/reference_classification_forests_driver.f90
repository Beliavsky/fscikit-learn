program reference_classification_forests_driver
  use sklearn_ensemble, only: random_forest_classifier,extra_trees_classifier
  use sklearn_common, only: dp
  use sklearn_ieee_math, only: ieee_total
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none
  class(random_forest_classifier), allocatable :: model
  logical :: score_ok
  integer, allocatable :: labels(:),map_labels(:),predicted(:),tree_classes(:)
  real(dp), allocatable :: proba(:,:),logproba(:,:),map_weights(:)
  integer :: cw,nmap,nclass
  integer :: cases,c,mode,n,p,m,criterion,depth,leaf,split,nf,seed,nw,i,t,nt,boot,ns,k,total,at,nrng
  integer, allocatable :: seeds(:),metadata(:),rng_before(:),rng_after(:),leaves(:,:)
  integer(int64) :: decrease_bits
  integer(int64), allocatable :: bits(:)
  real(dp), allocatable :: x(:,:),query(:,:),weights(:),values(:),impurities(:),mass(:),threshold(:)
  integer, allocatable :: left(:),right(:),features(:),count_(:)
  character(16), parameter :: criteria(4)=[character(16)::'gini','entropy','log_loss','unknown']
  character(20), parameter :: modes(4)=[character(20)::'none','balanced','balanced_subsample','unknown']
  read(*,*) cases
  do c=1,cases
    read(*,*) mode,n,p,m,criterion,depth,leaf,split,nf,seed,nw,decrease_bits,nt,boot,ns,cw,nmap
    allocate(x(n,p),query(m,p),bits(n*p));read(*,*) bits;x=reshape(transfer(bits,[0._dp],size(bits)),[n,p])
    deallocate(bits);allocate(labels(n));read(*,*) labels
    allocate(bits(m*p));read(*,*) bits;query=reshape(transfer(bits,[0._dp],size(bits)),[m,p]);deallocate(bits)
    if(nw>=0) then
      allocate(weights(nw),bits(nw));read(*,*) bits;weights=transfer(bits,weights);deallocate(bits)
    end if
    if(mode==0) then
      allocate(random_forest_classifier::model)
    else
      allocate(extra_trees_classifier::model)
    end if
    model%class_weight=modes(cw+1)
    if(nmap>=0) then
      allocate(model%class_weight_values);allocate(map_labels(nmap),map_weights(nmap),bits(nmap))
      read(*,*) map_labels;read(*,*) bits;map_weights=transfer(bits,map_weights)
      model%class_weight_values%labels=map_labels;model%class_weight_values%weights=map_weights
      deallocate(bits,map_labels,map_weights)
    end if
    model%criterion=criteria(criterion+1);model%max_depth=depth;model%min_samples_leaf=leaf
    model%min_samples_split=split;model%max_features=nf;model%random_state=seed
    model%min_impurity_decrease=transfer(decrease_bits,0._dp)
    model%n_estimators=nt;model%bootstrap=boot/=0;model%max_samples=ns
    call random_seed(size=nrng);allocate(rng_before(nrng),rng_after(nrng));call random_seed(get=rng_before)
    write(*,'(A,1X,I0)') 'BEGIN',c
    call model%fit(x,labels,weights)
    call random_seed(get=rng_after);call emit_i('rng_preserved',[merge(1,0,all(rng_before==rng_after))])
    predicted=model%predict(query);call emit_i('prediction',predicted)
    proba=model%predict_proba(query);logproba=model%predict_log_proba(query)
    call emit_r('proba',reshape(proba,[size(proba)]));call emit_r('logproba',reshape(logproba,[size(logproba)]))
    call emit_i('classes',model%classes_);nclass=model%n_classes_
    score_ok=.true.
    if(allocated(weights)) score_ok=score_ok.and.ieee_total(weights)/=0
    if(score_ok) call emit_r('score',[model%score(x,labels,weights)])
    leaves=model%apply(query);call emit_i('leaves',reshape(leaves,[size(leaves)]))
    call emit_r('importance',model%feature_importances_)
    allocate(seeds(nt),metadata(5*nt));total=0;k=0
    do t=1,nt
      seeds(t)=model%estimators_(t)%random_state;total=total+model%estimators_(t)%node_count_
      metadata(k+1:k+5)=[model%estimators_(t)%node_count_,model%estimators_(t)%get_depth(), &
        model%estimators_(t)%get_n_leaves(),model%estimators_(t)%n_features_in_,model%estimators_(t)%max_features_];k=k+5
    end do
    call emit_i('seeds',seeds);call emit_i('tree_metadata',metadata)
    call emit_i('forest_metadata',[model%n_features_in_,size(model%estimators_),model%n_classes_])
    allocate(tree_classes(nt*nclass));k=0
    do t=1,nt
      tree_classes(k+1:k+nclass)=model%estimators_(t)%classes_;k=k+nclass
    end do
    call emit_i('tree_classes',tree_classes)
    allocate(values(total*nclass),impurities(total),mass(total),threshold(total))
    allocate(left(total),right(total),features(total),count_(total));at=0
    do t=1,nt
      do i=1,model%estimators_(t)%node_count_
        at=at+1
        associate(node=>model%estimators_(t)%nodes_(i))
          values((at-1)*nclass+1:at*nclass)=node%probability;impurities(at)=node%impurity;mass(at)=node%weighted_n_samples;threshold(at)=node%threshold
          left(at)=node%left;right(at)=node%right;features(at)=node%feature;count_(at)=node%n_samples
        end associate
      end do
    end do
    call emit_r('values',values);call emit_r('impurity',impurities);call emit_r('mass',mass)
    call emit_r('threshold',threshold);call emit_i('left',left);call emit_i('right',right)
    call emit_i('feature',features);call emit_i('count',count_)
    call emit_i('criterion_preserved',[(merge(1,0,model%estimators_(t)%criterion==criteria(criterion+1)),t=1,nt)])
    deallocate(x,query,model,values,impurities,mass,threshold,left,right,features,count_,seeds,metadata,rng_before,rng_after,leaves,labels,tree_classes)
    if(allocated(weights)) deallocate(weights)
    write(*,'(A)') 'END'
  end do
contains
  subroutine emit_i(name,a)
    character(*), intent(in) :: name
    integer, intent(in) :: a(:)
    write(*,'(A,1X,I0,*(1X,I0))') name,size(a),a
  end subroutine
  subroutine emit_r(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(*,'(A,1X,I0,*(1X,ES26.17E3))') name,size(a),a
  end subroutine
end program
